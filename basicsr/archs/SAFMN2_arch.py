import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("/code/UHDformer-main")
from basicsr.archs.VAE_adapter import AutoencoderKL
import time
import yaml
from basicsr.utils.vae_util import instantiate_from_config
# from utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.utils.registry import ARCH_REGISTRY
import math



# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Local feature
class Local(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = int(dim // growth_rate)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.weight(y)
        return x*y


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True) #[1, 64, 64, 64]
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        self.local = Local(dim, ffn_scale)
        self.gobal = Gobal(dim)
        self.conv = nn.Conv2d(2*dim, dim, 1, 1, 0)
        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_l = self.local(y)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y_l, y_g], dim=1)) + x

        y = self.fc(self.norm2(y)) + y
        return y
    

class ResBlock(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, b=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return res + x

# @ARCH_REGISTRY.register()
class SAFMN2(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4,vae_weight=None):
        super().__init__()
        with open("/code/UHDformer-main/options/VAE/kl_8.yml") as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load(f, Loader=yaml.FullLoader)["network_g"]
            config.pop('type')
            self.vae = AutoencoderKL(**config)
            if vae_weight:
                
                msg = self.vae.load_state_dict(torch.load(vae_weight,map_location='cpu')["state_dict"],strict=False)
                print(f"load vae weight from {vae_weight}")
                print('missing keys:',len(msg.missing_keys),'unexpected keys:',len(msg.unexpected_keys))

                
        out_dim = 64

        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
            nn.PixelUnshuffle(upscaling_factor),
            nn.Conv2d(dim*upscaling_factor, dim, 1, 1, 0),
        )

        self.feats = nn.Sequential(*[AttBlock(out_dim, ffn_scale) for _ in range(n_blocks)])

        # self.to_img = nn.Sequential(
        #     nn.Conv2d(out_dim, dim, 3, 1, 1),
        #     nn.PixelShuffle(upscaling_factor)
        # )
        self.merge =  nn.Sequential(  # 这里考虑加一个MOE
            nn.Conv2d(out_dim, out_dim-dim, 3, 1, 1),
        )

    def forward(self, x):
        x0 = self.to_feat(x)
        x,adapters = self.vae.encode(x)
        x = torch.cat([x0, x], dim=1)
        x = self.feats(x) + x #[1, 64, 240, 135]
        x = self.merge(x)
        x = self.vae.decode(x,adapters)
        return x
    

    @torch.no_grad()
    def test_tile(self, input, tile_size=512, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height 
        output_width = width 
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self(input_tile)

                # output tile area on total image
                output_start_x = input_start_x 
                output_end_x = input_end_x 
                output_start_y = input_start_y 
                output_end_y = input_end_y 

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) 
                output_end_x_tile = output_start_x_tile + input_tile_width 
                output_start_y_tile = (input_start_y - input_start_y_pad) 
                output_end_y_tile = output_start_y_tile + input_tile_height 

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output
# for LOL dataset
# class SAFMN(nn.Module):
#     def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
#         super().__init__()
#         self.to_feat = nn.Sequential(
#             nn.Conv2d(3, dim, 3, 1, 1),
#             ResBlock(dim, 3, 1, 1)
#         )

#         self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

#         self.to_img = nn.Sequential(
#             ResBlock(dim, 3, 1, 1),
#             nn.Conv2d(dim, 3, 3, 1, 1)
#         )

#     def forward(self, x):
#         x = self.to_feat(x)
#         x = self.feats(x) + x
#         x = self.to_img(x)
#         return x

if __name__== '__main__': 
    x = torch.randn(1, 3, 1280, 1280).to('cuda')
    model = SAFMN2(dim=48, n_blocks=8, ffn_scale=2.0, upscaling_factor=8).to('cuda')
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"model parameters number:{para_num}") 
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        end_time = time.time()
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)