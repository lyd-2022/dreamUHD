import torch
import numbers
import math
import torch.nn.functional as F
import torch.nn as nn
# from basicsr.models.archs.arch_util import LayerNorm2d
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from scipy.signal.windows import gaussian
import numpy as np
import os
from torchvision import transforms

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


os.environ["CUDA_VISIBLE_DEVICES"]='0'



@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward_Restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)

class ConvNeXtBlockLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1
        )  # depthwise conv
        # self.norm = ConvNeXtBlockLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class CannyNet(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(CannyNet, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)

        return grad_mag

class MGF(nn.Module):
    def __init__(self, num_feature, kernel_size):
        super(MGF, self).__init__()

        self.num = kernel_size * kernel_size

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.d1 = default_conv(num_feature, num_feature, 3)
        self.g1 = default_conv(num_feature, num_feature, 3)

        self.depth_kernel = nn.Sequential(
            default_conv(num_feature, num_feature, 1),
            nn.ReLU(True),
            default_conv(num_feature, kernel_size ** 2, 1)
        )

        self.guide_kernel = nn.Sequential(
            default_conv(num_feature, num_feature, 1),
            nn.ReLU(True),
            default_conv(num_feature, kernel_size ** 2, 1)
        )

        self.d2 = default_conv(kernel_size ** 2, kernel_size ** 2, 1)
        self.g2 = default_conv(kernel_size ** 2, kernel_size ** 2, 1)

        self.d3 = default_conv(num_feature, num_feature, 3)
        self.g3 = default_conv(num_feature, num_feature, 3)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=1)
        self.inputs_conv = NAFBlock(num_feature) #ConvNeXtBlock(num_feature)

        self.guide_conv = NAFBlock(num_feature) #ConvNeXtBlock(num_feature)

    def getKernel(self, input_kernel):
        fuse_kernel = torch.tanh(input_kernel) / (
                self.aff_scale_const + 1e-8)
        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
        fuse_kernel = fuse_kernel / abs_kernel_sum

        return fuse_kernel

    def forward(self, depth, guide, S):
        b, c, h, w = depth.size()

        depth = self.d1(depth)
        guide = self.g1(guide)

        inputs_depth = self.inputs_conv(depth)
        guide_kernel = self.guide_kernel(guide)
        guide_kernel = self.g2(guide_kernel * S + guide_kernel)
        guide_kernel = self.getKernel(guide_kernel)
        unfold_inputs_depth = self.unfold(inputs_depth).view(b, c, -1, h, w)
        w_depth = torch.einsum('bkhw, bckhw->bchw', [guide_kernel, unfold_inputs_depth]) + inputs_depth

        inputs_guide = self.guide_conv(guide)
        depth_kernel = self.depth_kernel(depth)#w_depth
        depth_kernel = self.d2(depth_kernel * S + depth_kernel)
        depth_kernel = self.getKernel(depth_kernel)
        unfold_inputs_guide = self.unfold(inputs_guide).view(b, c, -1, h, w)
        w_guide = torch.einsum('bkhw, bckhw->bchw', [depth_kernel, unfold_inputs_guide]) + inputs_guide

        out_depth = self.d3(w_depth)
        out_guide = self.g3(w_guide)

        return out_depth, out_guide

class DPFI(nn.Module):
    def __init__(self, input_planes, weight_planes, scale_factor):
        super(DPFI, self).__init__()
        self.mean1 = nn.AvgPool2d(scale_factor, scale_factor)
        self.mean2 = nn.AvgPool2d(scale_factor, scale_factor)


        self.n1 = NAFBlock(input_planes)
        self.s1 = NAFBlock(input_planes)

        self.c1 = default_conv(input_planes, weight_planes, 3)
        self.c2 = default_conv(input_planes, weight_planes, 3)
        self.w = MGF(weight_planes, 3)

        self.up = Upsample(weight_planes,scale_factor)
        self.c3 = default_conv(weight_planes, weight_planes, 3)

    def forward(self, x, ns, seg):

        # Padding in case images are not multiples of 8
        h, w = ns.shape[2], ns.shape[3]
        if h!=w:
            factor = 8
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            ns_ = F.pad(ns, (0, padw, 0, padh), 'reflect')
            seg_ = F.pad(seg, (0, padw, 0, padh), 'reflect')

            ns = self.mean1(ns_)
            seg = self.mean2(seg_)
        else:
            ns = self.mean1(ns)
            seg = self.mean2(seg)
        ns1 = self.n1(ns)
        seg1 = self.s1(seg)

        ns_unfold_p = F.unfold(ns1, kernel_size=(3, 3), padding=1)

        seg_unfold = F.unfold(seg1, kernel_size=(3, 3), padding=1)
        seg_unfold_p = seg_unfold.permute(0, 2, 1).contiguous()

        ns_unfold_p = F.normalize(ns_unfold_p, dim=1)
        seg_unfold_p = F.normalize(seg_unfold_p, dim=2)

        R_dn = torch.bmm(seg_unfold_p, ns_unfold_p)
        Score_dn = torch.diagonal(R_dn, dim1=-2, dim2=-1)

        S_ns = Score_dn.view(Score_dn.size(0), 1, ns.size(2), ns.size(3))##w

        seg = self.c1(seg)
        ns = self.c2(ns)
        out_seg, out_ns = self.w(seg, ns, S_ns)
        out = self.c3(self.up(out_seg + out_ns))
        out = out[:, :, :h, :w] + x
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_q, x_kv):
        '''
        input:
            x_q: for query
            x_kv: for key and value
        normally:
            feature is x_q, and prompt is x_kv
            this imple that feature will select the prompt
        cross attention as no skip connection in default
        '''
        b, c, h, w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class SPFI(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SPFI, self).__init__()
        self.norm_m= LayerNorm(dim, LayerNorm_type)
        self.norm_n = LayerNorm(dim, LayerNorm_type)
        self.atten1 = Cross_Attention(dim, num_heads, bias)
        self.atten2 = Cross_Attention(dim, num_heads, bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward_Restormer(dim, ffn_expansion_factor, bias)
        self.ffn2 = FeedForward_Restormer(dim, ffn_expansion_factor, bias)

    def forward(self, x, m, n):
        m = m + self.atten1(self.norm_m(m), self.norm1(x))
        m = m + self.ffn1(self.norm2(m))

        n = n + self.atten2(self.norm_n(n), self.norm1(x))
        n = n + self.ffn2(self.norm3(n))

        return x, m, n

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Net(nn.Module):
    def __init__(self, channel_query_dict, scale_factor=8,  num_blocks=4, num_heads=8, ffn_expansion_factor=2.66, bias=True,
                 LayerNorm_type='WithBias'):
        super().__init__()
        self.channel_query_dict = channel_query_dict
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))
        self.grad = CannyNet(threshold=3.0, use_cuda=True)

        self.high0 = nn.Sequential(NAFBlock(channel_query_dict[256]),
                                   NAFBlock(channel_query_dict[256]))
        self.high1 = nn.Sequential(NAFBlock(channel_query_dict[256]),
                                   NAFBlock(channel_query_dict[256]))
        self.high2 = nn.Sequential(NAFBlock(channel_query_dict[256]),
                                   NAFBlock(channel_query_dict[256]))

        self.down0 = Downsample(channel_query_dict[256], scale_factor)
        self.down1 = Downsample(channel_query_dict[256], scale_factor)
        self.down2 = Downsample(channel_query_dict[256], scale_factor)

        self.conv1= nn.Sequential(nn.Conv2d(1, channel_query_dict[256], 3, 1, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))

        self.down_g= Downsample(channel_query_dict[256], scale_factor)
        self.down_n = Downsample(channel_query_dict[256], scale_factor)

        self.shallow1 = NAFBlock(channel_query_dict[256])
        self.shallow2 = NAFBlock(channel_query_dict[256])
        self.middle1 = NAFBlock(channel_query_dict[256])
        self.middle2 = NAFBlock(channel_query_dict[256])
        self.deep1 = NAFBlock(channel_query_dict[256])
        self.deep2 = NAFBlock(channel_query_dict[256])

        self.spf1= SPFI(dim=int(channel_query_dict[256]), num_heads=num_heads,
                                ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.spf2 = SPFI(dim=int(channel_query_dict[256]), num_heads=num_heads,
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.spf3 = SPFI(dim=int(channel_query_dict[256]), num_heads=num_heads,
                         ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.dpf1 = DPFI(input_planes=channel_query_dict[256], weight_planes=channel_query_dict[256], scale_factor=scale_factor//2
                           )
        self.dpf2 = DPFI(input_planes=channel_query_dict[256], weight_planes=channel_query_dict[256],
                      scale_factor=scale_factor//2,
                      )
        self.dpf3 = DPFI(input_planes=channel_query_dict[256], weight_planes=channel_query_dict[256],
                      scale_factor=scale_factor//2,
                      )

        self.TB1 = nn.Sequential(*[
            NAFBlock(channel_query_dict[256]) for i in range(num_blocks)])
        self.TB2 = nn.Sequential(*[
            NAFBlock(channel_query_dict[256]) for i in range(num_blocks)])
        self.TB3 = nn.Sequential(*[
            NAFBlock(channel_query_dict[256]) for i in range(num_blocks)])
        self.x_up1 = Upsample(channel_query_dict[256], scale_factor)
        self.x_up2 = Upsample(channel_query_dict[256], scale_factor)
        self.x_up3 = Upsample(channel_query_dict[256], scale_factor)

        self.fusion1 = nn.Conv2d(channel_query_dict[256] * 2, channel_query_dict[256], 1)
        self.fusion2 = nn.Conv2d(channel_query_dict[256] * 2, channel_query_dict[256], 1)
        self.fusion3 = nn.Conv2d(channel_query_dict[256] * 2, channel_query_dict[256], 1)
        self.out1 = nn.Sequential(NAFBlock(channel_query_dict[256]),
                                  NAFBlock(channel_query_dict[256]),
                                  nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))
        self.out = nn.Sequential(NAFBlock(channel_query_dict[256]),
                                 NAFBlock(channel_query_dict[256]),
                                 NAFBlock(channel_query_dict[256]),
                                 nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))

    def forward(self, x, ns):

        ori = x
        img_grad = self.grad(x)
        enter = self.enter(x)

        x1 = self.high0(enter)
        x1_in = self.down0(x1)
        grad_in = self.down_g(self.conv1(img_grad.data))
        grad1 = self.shallow1(grad_in)
        ns_in = self.down_n(self.conv2(ns))
        ns1 = self.shallow2(ns_in)
        out_x1, out_grad1, out_ns1 = self.spf1(x1_in, grad1, ns1)
        out_x1 = self.dpf1(out_x1, out_grad1, out_ns1)
        out_x1 = self.TB1(out_x1)
        x2 = self.fusion1(torch.cat([x1, self.x_up1(out_x1)], dim=1))

        x2 = self.high1(x2)
        x2_in =self.down1(x2)
        grad2 = self.middle1(out_grad1)
        ns2 = self.middle2(out_ns1)
        out_x2, out_grad2, out_ns2 = self.spf2(x2_in, grad2, ns2)
        out_x2 = self.dpf2(out_x2, out_grad2, out_ns2)
        out_x2 = self.TB2(out_x2)
        x3 = self.fusion2(torch.cat([x2, self.x_up2(out_x2)], dim=1))

        x3 = self.high2(x3)
        x3_in = self.down2(x3)
        grad3 = self.deep1(out_grad2)
        ns3 = self.deep2(out_ns2)
        out_x3, out_grad3, out_ns3 = self.spf3(x3_in, grad3, ns3)
        out_x3 = self.dpf3(out_x3, out_grad3, out_ns3)
        out_x3 = self.TB3(out_x3)
        fusion = out_x3 + out_grad3 + out_ns3

        fusion = self.x_up3(fusion)

        out1 = self.out1(fusion)
        out1 = out1 + ori

        fusion = self.fusion3(torch.cat([x3, fusion], dim=1))
        out = self.out(fusion) + ori

        return out, out1


class UHDDIP(nn.Module):
    def __init__(self,
                 *,
                 scale_factor=8,
                 num_blocks=4,
                 bias=True,
                 LayerNorm_type='WithBias'
                 ):
        super().__init__()
        channel_query_dict = {8: 256, 16: 256, 32: 384, 64: 192, 128: 96, 256: 16, 512: 32}
        self.restoration_network = Net(channel_query_dict=channel_query_dict,
                                       scale_factor=scale_factor,
                                       num_blocks=num_blocks,
                                       bias=bias,
                                       LayerNorm_type=LayerNorm_type)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, normal):
        restoration, out1= self.restoration_network(input, normal)
        return restoration , out1

    def test(self, input, normal):
        _, _, h_old, w_old = input.shape
        restoration, out1= self.encode_and_decode(input, normal)
        return restoration, out1

    def forward(self, input, normal):
        restoration, out1 = self.encode_and_decode(input, normal)
        return restoration, out1

def main():

    x = torch.rand(1, 3, 256, 256)#512, 512
    n = torch.rand(1, 3, 256, 256)
    model = UHDDIP()
    # print(model)
    y = model(x, n)
    # print(y.shape)
    from thop import profile
    from thop import clever_format
    # input = torch.randn(1,3,256,256)


    # clean = torch.randn(1,3,256,256)
    flops,params = profile(model,inputs=(x,n))
    flops,params = clever_format([flops,params], "%.3f")
    print(f"params:{params},flops:{flops}")

if __name__ == '__main__':
    main()