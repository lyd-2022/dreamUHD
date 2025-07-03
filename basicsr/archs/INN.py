import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import sys
sys.path.append('/mnt/petrelfs/liuyidi/Code/dreamUHD')
from basicsr.archs.Resblock.Res_four import Res_four10
logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self.logdet = logdet  # 默认是否计算logdet

    def initialize(self, input):
        with torch.no_grad():
            mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)
            std = torch.std(input, dim=(0, 2, 3), keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, compute_logdet=True):
        # 如果不想计算logdet，可以覆盖 self.logdet
        compute_logdet = compute_logdet and self.logdet

        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        out = self.scale * (input + self.loc)

        if compute_logdet:
            # 计算本层对数行列式
            log_abs = logabs(self.scale)  # shape: (1, C, 1, 1)
            logdet = height * width * torch.sum(log_abs)
            return out, logdet
        else:
            return out, None

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input, compute_logdet=True):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        if compute_logdet:
            logdet = height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        else:
            logdet = None
        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ (
                (self.w_u * self.u_mask)
                + torch.diag(self.s_sign * torch.exp(self.w_s))
            )
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, input, compute_logdet=True):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        if compute_logdet:
            logdet = height * width * torch.sum(self.w_s)
        else:
            logdet = None
        return out, logdet

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out
    
class ZerodepConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, padding=0,groups=in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = self.conv2(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=16, affine=True):
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, in_channel // 2, 3, padding=1,groups=in_channel//2),
            nn.Conv2d(in_channel // 2, filter_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[1].weight.data.normal_(0, 0.05)
        self.net[1].bias.data.zero_()
        self.net[3].weight.data.normal_(0, 0.05)
        self.net[3].bias.data.zero_()

    def forward(self, input, compute_logdet=True):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            if compute_logdet:
                logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
            else:
                logdet = None
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        out = torch.cat([in_a, out_b], 1)
        return out, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input, compute_logdet=True):
        out, logdet = self.actnorm(input, compute_logdet=compute_logdet)
        out, det1 = self.invconv(out, compute_logdet=compute_logdet)
        out, det2 = self.coupling(out, compute_logdet=compute_logdet)

        if compute_logdet:
            if det1 is not None:
                logdet = logdet + det1
            if det2 is not None:
                logdet = logdet + det2
        else:
            logdet = None
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True,compute_logdet=False):
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))
        self.split = split
        if compute_logdet:
            if split:
                self.prior = ZerodepConv2d(in_channel * 2, in_channel * 4)
            else:
                self.prior = ZerodepConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, compute_logdet=True):
        b_size, n_channel, height, width = input.shape
        # Squeeze
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0.0 if compute_logdet else None

        for flow in self.flows:
            out, det = flow(out, compute_logdet=compute_logdet)
            if compute_logdet and det is not None:
                logdet += det

        if self.split:
            out, z_new = out.chunk(2, 1)
            
            if compute_logdet:
                mean, log_sd = self.prior(out).chunk(2, 1)
                log_p = gaussian_log_p(z_new, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
            else:
                log_p = None
        else:
            
            
            if compute_logdet:
                zero = torch.zeros_like(out)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                log_p = gaussian_log_p(out, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
            else:
                log_p = None
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=True):
        input = output
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input, compute_logdet=False):
        log_p_sum = 0.0 if compute_logdet else None
        logdet = 0.0 if compute_logdet else None
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, compute_logdet=compute_logdet)
            z_outs.append(z_new)

            if compute_logdet:
                if det is not None:
                    logdet += det
                if log_p is not None:
                    log_p_sum += log_p
            
        # 如果不需要logdet，可只返回变换后的结果
        if compute_logdet:
            return log_p_sum, logdet, z_outs
        else:
            for i,z in enumerate(z_outs[::-1]):  
                
                
                if i == 0:                   
                    out = z
                else:
                    out = torch.cat([out, z], dim=1)
                b_size, c, h, w = out.shape  # 获取当前的 out 形状
                out = out.view(b_size, c // 4, 2, 2, h, w)  # 逆 Squeeze
                out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
                out = out.view(b_size, c // 4, h * 2, w * 2)  # 还原到原始空间




            return out

    def reverse(self, z_list, reconstruct=True):
        # 逆向变换不依赖 logdet，这里无需额外参数
        input = None
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input


# ============= 测试示例 =============
if __name__ == '__main__':
    # 构造Glow模型
    model = Glow(in_channel=64, n_flow=2, n_block=1, affine=True, conv_lu=True)

    # 生成随机输入
    # batch_size, C, H, W = 2, 32, 128, 128
    # x = torch.randn(batch_size, C, H, W)

    # # --- 1) 启用logdet计算 ---
    # log_p_sum, logdet, z_outs = model(x, compute_logdet=True)
    # print(f"With logdet:")
    # print(f"  log_p_sum: {log_p_sum}")
    # print(f"  logdet   : {logdet}")

    # # 逆向变换
    # x_recon = model.reverse(z_outs, reconstruct=False)
    # mse_error = torch.mean((x - x_recon) ** 2).item()
    # print(f"  MSE Error: {mse_error:.8e}\n")

    # # --- 2) 禁用logdet计算 ---
    # z_outs_no = model(x, compute_logdet=False)
    # print(f"Without logdet:")

    # # 逆向变换（依旧可以重建输入，但不会有对数似然和行列式信息）
    # x_recon_no = model.reverse(z_outs_no[0], reconstruct=False)
    # mse_error_no = torch.mean((x - x_recon_no) ** 2).item()
    # print(f"  MSE Error: {mse_error_no:.8e}")

    # 分析参数量以M为单位
    #遍历模型中每个模块，打印每一个模块的参数量

    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params/1e6:.2f}M")
    
    # num_params = sum(p.numel() for p in model.parameters())
    # # print(f"  Number of parameters: {num_params}")
    # print(f"  Number of parameters: {num_params/1e6:.2f}M")



    