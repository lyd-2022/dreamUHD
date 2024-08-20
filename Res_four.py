import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
def Normalize(in_channels, num_groups=4):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class Res_four9(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.splitconv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.splitconv2 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv1 = DepthwiseSeparableConv(out_channels//2,
                                     out_channels//2,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels//2)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = DepthwiseSeparableConv(out_channels//2,
                                     out_channels//2,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.mergeconv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.ampconv0 = nn.Sequential(
            nn.Conv2d(in_channels//2+1, in_channels//2+1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(in_channels//2+1, in_channels//2+1, kernel_size=1, stride=1, padding=0)
        )
        # self.phaconv0 = nn.Sequential(
        #     nn.Conv2d(in_channels//2+1, in_channels//2+1, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.1,inplace=True),
        #     nn.Conv2d(in_channels//2+1, in_channels//2+1, kernel_size=1, stride=1, padding=0)
        # )
        self.ampconv = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1, stride=1, padding=0))
        

        # self.phaconv = nn.Sequential(
        #     nn.Conv2d(in_channels//2, out_channels//2, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.1,inplace=True),
        #     nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1, stride=1, padding=0))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):

        h = torch.fft.rfft(x, dim=1)
        amp, phase = torch.abs(h), torch.angle(h)
        amp = self.ampconv0(amp)
        # phase = self.phaconv0(phase)
        h = torch.complex(amp*torch.cos(phase),amp*torch.sin(phase))
        h = torch.fft.irfft(h, dim=1, n=x.size(1))
        h = x+h
        h = self.norm1(h)
        x1= self.splitconv1(h)
        x11= torch.fft.rfft2(x1, norm='backward')
        amp, phase = torch.abs(x11), torch.angle(x11)
        amp = self.ampconv(amp)
        # phase = self.phaconv(phase)
        x11 = torch.complex(amp*torch.cos(phase),amp*torch.sin(phase))
        x11 = torch.fft.irfft2(x11, s=(h.shape[2],h.shape[3]),norm='backward')
        x1 = x1+x11

        h = self.splitconv2(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        space_score = nn.Sigmoid()(x1)
        fre_score = nn.Sigmoid()(h)
        h = h*space_score
        x1 = x1*fre_score
        h = torch.cat([x1,h],dim=1)
        h = self.mergeconv(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
