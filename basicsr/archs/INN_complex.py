import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import sys
sys.path.append("/mnt/petrelfs/liuyidi/Code/BasicSR-master")
from basicsr.archs.INN import Block

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

###############################################
# 辅助函数：针对复数数据的 squeeze / unsqueeze
###############################################
def complex_squeeze(x):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    x = x.view(B, C, H//2, 2, W//2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * 4, H // 2, W // 2)
    return x

def complex_unsqueeze(x):
    # x: (B, C, H, W)，要求 C 能被4整除
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // 4, H * 2, W * 2)
    return x

###############################################
# 复数 ActNorm 层
###############################################
class ComplexActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.loc_r = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.loc_i = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale_r = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.scale_i = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.initialized = False

    def initialize(self, x_r, x_i):
        with torch.no_grad():
            mean_r = torch.mean(x_r, dim=(0,2,3), keepdim=True)
            mean_i = torch.mean(x_i, dim=(0,2,3), keepdim=True)
            std_r = torch.std(x_r, dim=(0,2,3), keepdim=True)
            std_i = torch.std(x_i, dim=(0,2,3), keepdim=True)
            self.loc_r.data.copy_(-mean_r)
            self.loc_i.data.copy_(-mean_i)
            self.scale_r.data.copy_(1/(std_r+1e-6))
            self.scale_i.data.copy_(1/(std_i+1e-6))
            self.initialized = True

    def forward(self, x_r, x_i):
        if not self.initialized:
            self.initialize(x_r, x_i)
        out_r = self.scale_r * x_r - self.scale_i * x_i + self.loc_r
        out_i = self.scale_i * x_r + self.scale_r * x_i + self.loc_i
        return out_r, out_i

    def reverse(self, y_r, y_i):
        denom = self.scale_r**2 + self.scale_i**2 + 1e-6
        x_r = ( self.scale_r*(y_r - self.loc_r) + self.scale_i*(y_i - self.loc_i) )/denom
        x_i = (-self.scale_i*(y_r - self.loc_r) + self.scale_r*(y_i - self.loc_i) )/denom
        return x_r, x_i

###############################################
# 复数 1×1 可逆卷积层（权重初始化为单位矩阵）
###############################################
class ComplexInvConv2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        weight = torch.eye(in_channels).view(in_channels, in_channels, 1, 1)
        self.weight_r = nn.Parameter(weight.clone())
        self.weight_i = nn.Parameter(torch.zeros_like(weight))

    def forward(self, x_r, x_i):
        out_r = F.conv2d(x_r, self.weight_r) - F.conv2d(x_i, self.weight_i)
        out_i = F.conv2d(x_i, self.weight_r) + F.conv2d(x_r, self.weight_i)
        return out_r, out_i

    def reverse(self, y_r, y_i):
        weight_complex = self.weight_r.squeeze() + 1j * self.weight_i.squeeze()  # (C, C)
        inv_weight = torch.linalg.inv(weight_complex)
        inv_weight = inv_weight.unsqueeze(-1).unsqueeze(-1)
        inv_r = inv_weight.real
        inv_i = inv_weight.imag
        x_r = F.conv2d(y_r, inv_r) - F.conv2d(y_i, inv_i)
        x_i = F.conv2d(y_i, inv_r) + F.conv2d(y_r, inv_i)
        return x_r, x_i

###############################################
# 用于耦合层中的辅助实数 ZerodepConv2d（与原代码一致）
###############################################
class ZerodepConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, padding=0, groups=in_channel)
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

###############################################
# 复数 仿射耦合层
###############################################
class ComplexAffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=16, affine=True):
        super().__init__()
        self.affine = affine
        # 将通道数分为两部分（对于复数，每部分包含实部和虚部）
        in_split = in_channel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_split * 2, in_split, kernel_size=3, padding=1, groups=in_split),
            nn.Conv2d(in_split, filter_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, kernel_size=1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[1].weight.data.normal_(0, 0.05)
        self.net[1].bias.data.zero_()
        self.net[3].weight.data.normal_(0, 0.05)
        self.net[3].bias.data.zero_()

    def forward(self, x_r, x_i, compute_logdet=True):
        in_split = x_r.shape[1] // 2
        a_r, b_r = x_r[:, :in_split, :, :], x_r[:, in_split:, :, :]
        a_i, b_i = x_i[:, :in_split, :, :], x_i[:, in_split:, :, :]
        a_cat = torch.cat([a_r, a_i], dim=1)
        net_out = self.net(a_cat)
        s, t = net_out.chunk(2, dim=1)
        s = torch.sigmoid(s + 2.0)
        b_r_out = (b_r + t) * s
        b_i_out = (b_i + t) * s
        out_r = torch.cat([a_r, b_r_out], dim=1)
        out_i = torch.cat([a_i, b_i_out], dim=1)
        if compute_logdet:
            logdet = torch.sum(torch.log(s).view(x_r.shape[0], -1), dim=1)
        else:
            logdet = None
        return out_r, out_i, logdet

    def reverse(self, y_r, y_i):
        in_split = y_r.shape[1] // 2
        a_r, b_r = y_r[:, :in_split, :, :], y_r[:, in_split:, :, :]
        a_i, b_i = y_i[:, :in_split, :, :], y_i[:, in_split:, :, :]
        a_cat = torch.cat([a_r, a_i], dim=1)
        net_out = self.net(a_cat)
        s, t = net_out.chunk(2, dim=1)
        s = torch.sigmoid(s + 2.0)
        b_r_in = b_r / s - t
        b_i_in = b_i / s - t
        x_r = torch.cat([a_r, b_r_in], dim=1)
        x_i = torch.cat([a_i, b_i_in], dim=1)
        return x_r, x_i

###############################################
# 复数 Flow 层：依次执行 ActNorm -> 1×1 可逆卷积 -> 仿射耦合
###############################################
class ComplexFlow(nn.Module):
    def __init__(self, in_channel, affine=True):
        super().__init__()
        self.actnorm = ComplexActNorm(in_channel)
        self.invconv = ComplexInvConv2d(in_channel)
        self.coupling = ComplexAffineCoupling(in_channel, affine=affine)

    def forward(self, x_r, x_i, compute_logdet=True):
        x_r, x_i = self.actnorm(x_r, x_i)
        x_r, x_i = self.invconv(x_r, x_i)
        x_r, x_i, logdet = self.coupling(x_r, x_i, compute_logdet=compute_logdet)
        return x_r, x_i, logdet

    def reverse(self, x_r, x_i):
        x_r, x_i = self.coupling.reverse(x_r, x_i)
        x_r, x_i = self.invconv.reverse(x_r, x_i)
        x_r, x_i = self.actnorm.reverse(x_r, x_i)
        return x_r, x_i

###############################################
# 复数 Block 层（增加 split 多尺度）
# 每个 Block 内先进行 squeeze，再堆叠多个 Flow 层，
# 若 split=True，则在最后将特征通道一分为二：一部分作为隐变量 z，
# 另一部分作为下一尺度的输入。
###############################################
class ComplexBlock(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True,compute_logdet=False):
        super().__init__()
        self.split = split
        # squeeze 后通道数为 in_channel*4
        self.flows = nn.ModuleList()
        squeeze_channels = in_channel * 4
        for i in range(n_flow):
            self.flows.append(ComplexFlow(squeeze_channels, affine=affine))
        if compute_logdet:
            if self.split:
                self.prior = ZerodepConv2d(in_channel * 2, in_channel * 4)
            else:
                self.prior = ZerodepConv2d(squeeze_channels, squeeze_channels * 2)

    def forward(self, x_r, x_i, compute_logdet=True):
        # squeeze（对实部和虚部分别操作）
        x_r = complex_squeeze(x_r)
        x_i = complex_squeeze(x_i)
        logdet_total = 0.0 if compute_logdet else None
        for flow in self.flows:
            x_r, x_i, logdet = flow(x_r, x_i, compute_logdet=compute_logdet)
            if compute_logdet:
                logdet_total += logdet
        if self.split:
            # 按通道分为两部分：remain 和隐变量 z
            remain_r, z_r = x_r.chunk(2, dim=1)
            remain_i, z_i = x_i.chunk(2, dim=1)
            return remain_r, remain_i, logdet_total, torch.complex(z_r, z_i)

        else:
            return x_r, x_i, logdet_total, None

    def reverse(self, x_r, x_i, z=None, reconstruct=True):
        if self.split:
            if reconstruct:
                x_r = torch.cat([x_r, z.real], dim=1)
                x_i = torch.cat([x_i, z.imag], dim=1)
            else:
                # 非重构模式下可从先验采样（此处不展开）
                pass
        for flow in self.flows[::-1]:
            x_r, x_i = flow.reverse(x_r, x_i)
        # unsqueeze 恢复空间尺寸
        x_r = complex_unsqueeze(x_r)
        x_i = complex_unsqueeze(x_i)
        return torch.complex(x_r, x_i)




###############################################
# 复数 Glow 模型（多尺度 split 结构）
# 注意：每个 Block（除最后一层）采用 split 后，其输出通道数变为原 in_channel*2，
# 所以下一 Block 的输入通道数应相应更新。
###############################################
class MutiComplexGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        # 前 n_block-1 个 Block 均使用 split，输出通道数翻倍
        self.blocks.append(Block(in_channel, n_flow, split=True, affine=affine))
        for i in range(n_block - 2):
            in_channel = in_channel * 2
            self.blocks.append(ComplexBlock(in_channel, n_flow, split=True, affine=affine))
              # 更新下一个 Block 的输入通道数
        # 最后一个 Block 不做 split
        in_channel = in_channel * 2
        self.blocks.append(ComplexBlock(in_channel, n_flow, split=False, affine=affine))

    def forward(self, scale, compute_logdet=False):
        logdet_total = 0.0 if compute_logdet else None
        latents = []  # 存储每个 Block 的隐变量
        for i,block in enumerate(self.blocks):
            if i == 0:
                last, logdet, log_p, z = block(scale[i], compute_logdet=compute_logdet)
                latents.append(z)
            else:
                out, last, logdet, z = block(last, scale[i], compute_logdet=compute_logdet)


                latents.append(z)
                latents.append(out)
                if i == len(self.blocks) - 1:
                    latents.append(last)

            if compute_logdet and logdet is not None:
                logdet_total += logdet
        out = [complex_unsqueeze(latents[-1]),complex_unsqueeze(torch.cat([latents[1].imag,complex_unsqueeze(latents[-2])],dim=1)), complex_unsqueeze(torch.cat([latents[0],complex_unsqueeze(torch.cat([latents[1].real,latents[2]],dim=1))] ,dim=1))]
        #反转列表
        out = out[::-1]

        if compute_logdet:
            return out, latents, logdet_total
        else:
            return out

    def reverse(self,  scale,reconstruct=True):
        f11,f12= complex_squeeze(scale[0]).chunk(2, dim=1)
        f121,f122 = complex_squeeze(f12).chunk(2, dim=1)
        f21,f22 = complex_squeeze(scale[1]).chunk(2, dim=1)
        f3 = complex_squeeze(scale[2])
        c22 = torch.complex(f122,f22)
        c21 = torch.complex(f121,f21)
        # input3=torch.complex(f22,f3)
        c3 = self.blocks[-1].reverse(complex_squeeze(f22),f3)
        # scale[2] = c3.imag
        f22 = c3.real
        output2 = self.blocks[-2].reverse(f122,f22,c21)
        # scale[1] = output2.imag
        f12= output2.real
        # scale[0] = self.blocks[0].reverse(f12,f11)
        out = [self.blocks[0].reverse(f12,f11),output2.imag,c3.imag]

        return out



class ComplexGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        # 前 n_block-1 个 Block 均使用 split，输出通道数翻倍
        for i in range(n_block - 1):
            self.blocks.append(ComplexBlock(in_channel, n_flow, split=True, affine=affine))
            in_channel = in_channel * 2  # 更新下一个 Block 的输入通道数
        # 最后一个 Block 不做 split
        self.blocks.append(ComplexBlock(in_channel, n_flow, split=False, affine=affine))

    def forward(self, x_r, x_i, compute_logdet=True):
        logdet_total = 0.0 if compute_logdet else None
        latents = []  # 存储每个 Block 的隐变量
        for block in self.blocks:
            x_r, x_i, logdet, z = block(x_r, x_i, compute_logdet=compute_logdet)
            if compute_logdet and logdet is not None:
                logdet_total += logdet
            if block.split:
                latents.append(z)
            else:
                latents.append(torch.complex(x_r, x_i))
        # 最后一个 Block 的输出作为最终 latent

        final_latent = torch.complex(x_r, x_i)



        for i,z in enumerate(latents[::-1]):


            if i == 0:
                final_latent = z
            else:
                final_latent = torch.cat([final_latent, z], dim=1)

            final_latent =  complex_unsqueeze(final_latent)

        if compute_logdet:
            return final_latent, latents, logdet_total
        else:
            return final_latent

    def reverse(self,  latents,reconstruct=True):

        input = None
        #复数转换为实部和虚部
        x_real, x_imag = latents[-1].real, latents[-1].imag

        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(latents[-1].real, latents[-1].imag, latents[-1],reconstruct=reconstruct)
            else:
                input = block.reverse(input.real,input.imag, latents[-(i + 1)], reconstruct=reconstruct)
        return input


###############################################
# 测试示例：验证复数 Glow 模型（含多尺度 split）的可逆性
###############################################
if __name__ == '__main__':
    torch.manual_seed(1)
    batch_size = 2
    channels = 8    # 初始输入通道数
    height, width = 32, 32
    # 构造随机复数输入：实部和虚部分别为 tensor
    # x_real = torch.randn(batch_size, channels, height, width)
    # x_imag = torch.randn(batch_size, channels, height, width)

    scale = [torch.randn(batch_size, channels * 2**i, height // 2 ** i, width // 2 ** i) for i in range(3)]

    # 构造复数 Glow 模型：例如每个 Block 内含 2 个 Flow，共 2 个 Block（前一个 Block使用 split，多尺度）
    # model = ComplexGlow(in_channel=channels, n_flow=2, n_block=1, affine=True)
    # #验证可逆性
    # # 前向传播：得到最终 latent、各 Block 隐变量及 logdet
    # final_latent,latents,logdet = model(x_real, x_imag, compute_logdet=True)
    # # 逆向传播：利用 final_latent 和各尺度隐变量还原输入
    # x_recon = model.reverse(latents,reconstruct=True)
    # # 计算重构误差
    # mse_loss = torch.mean((final_latent.real - x_real)**2 + (final_latent.imag - x_imag)**2).item()
    # print(f"  MSE Error: {mse_loss :.8e}")
    model = MutiComplexGlow(in_channel=channels, n_flow=2, n_block=3, affine=True)

    # 前向传播：得到最终 latent、各 Block 隐变量及 logdet
    scale2,latents = model(scale, compute_logdet=False)
    # scale3 =scale2
    # # 逆向传播：利用 final_latent 和各尺度隐变量还原输入
    x_recon = model.reverse( scale2,latents)

    mse_loss1 = torch.mean((x_recon[0] - scale[0])** 2 ).item()
    print(f"  MSE Error1: {mse_loss1 :.8e}")
    mse_loss2 = torch.mean((x_recon[1] - scale[1])** 2).item()
    print(f"  MSE Error2: {mse_loss2 :.8e}")
    mse_loss3 = torch.mean((x_recon[2] - scale[2])** 2).item()
    print(f"  MSE Error3: {mse_loss3 :.8e}")
