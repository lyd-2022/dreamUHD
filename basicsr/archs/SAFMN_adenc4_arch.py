import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("/code/UHDformer-main")
from basicsr.archs.VAE_arch import AutoencoderKL
import time
import yaml
from basicsr.utils.vae_util import instantiate_from_config
# from utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.utils.registry import ARCH_REGISTRY
import math
from basicsr.utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.archs.encoder import nonlinearity, Normalize, ResnetBlock, make_attn, Downsample, Upsample
from basicsr.archs.wtconv import WTConv2d
import numpy as np  
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

@ARCH_REGISTRY.register()
class SAFMN_adenc4(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4,vae_weight=None,config=None,sample= True):
        super().__init__()
        with open(config) as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load(f, Loader=yaml.FullLoader)["network_g"]
            config.pop('type')
            self.vae = AutoencoderKL(**config)
            self.sample = sample
            if vae_weight:
                
                msg = self.vae.load_state_dict(torch.load(vae_weight,map_location='cpu')["params"],strict=False)
                print(f"load vae weight from {vae_weight}")
                print('missing keys:',len(msg.missing_keys),'unexpected keys:',len(msg.unexpected_keys))
                

            for name, param in self.vae.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
                
        out_dim = 80

        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
            nn.PixelUnshuffle(upscaling_factor),
            nn.Conv2d(3*upscaling_factor**2, dim, 1, 1, 0),
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
        x = self.vae.encode(x)
       

        
        x = torch.cat([x0, x], dim=1)
        x = self.feats(x) + x #[1, 64, 240, 135]
        x = self.merge(x)
        posterior = DiagonalGaussianDistribution(x)
        if self.sample:
            x = posterior.sample()
        else:
            x = posterior.mode()

        
        x = self.vae.decode(x)

        

        
        return x, posterior
    

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
class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,    
                 embed_dim,
                 optim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.learning_rate = optim["lr"]
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        
        return moments

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        #采样过程合并到encode中
        z,posterior = self.encode(input)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        # x = batch[k]
        x = batch
        # print(x.shape)
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
        
        

        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, in_channels)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, height*width, channels)
        x_flat = self.fc1(x_flat)
        x_flat = self.relu(x_flat)
        x_flat = self.fc2(x_flat)
        x_flat = x_flat.permute(0, 2, 1).view(batch_size, channels, height, width)  # Reshape back
        return x + x_flat
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.adapter_wtconv1 = WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.adapter_wtconv2 = WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.adapter_wtconv3 = WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.adapter_wtconv4 = WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.adapter_wtconv5 = WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            adapters = nn.ModuleList()  # Adapters for each block
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                adapters.append(Adapter(block_out, block_out // 4))  # Add Adapter
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            down.adapters = adapters  # Add adapters to down
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.adapter = Adapter(block_in, block_in // 4)  # Add Adapter

        self.shuffle1 = nn.PixelUnshuffle(2)
        self.shuffle2 = nn.PixelUnshuffle(4)
        self.shuffle3 = nn.PixelUnshuffle(8)
        self.shuffle4 = nn.PixelUnshuffle(16)
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.adapter_merge = nn.ModuleList([
            torch.nn.Conv2d(3+32, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(32+12, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(64+48, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(64+192, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(128+768, 128, kernel_size=1, stride=1, padding=0),
        ])
        
        

    def forward(self, x):
        # timestep embedding
        temb = None
        adapter_in1 = self.adapter_wtconv1(x)
        adapter_in2 = self.adapter_wtconv2(adapter_in1)
        adapter_in3 = self.adapter_wtconv3(adapter_in2)
        adapter_in4 = self.adapter_wtconv4(adapter_in3)
        adapter_in5 = self.adapter_wtconv5(adapter_in4)

        adapter_in2 = self.shuffle1(adapter_in2)
        adapter_in3 = self.shuffle2(adapter_in3)
        adapter_in4 = self.shuffle3(adapter_in4)
        adapter_in5 = self.shuffle4(adapter_in5)
        adapter_in = [adapter_in1, adapter_in2, adapter_in3, adapter_in4,adapter_in5]
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].adapters[i_block](h)  # Apply Adapter
                h = self.adapter_merge[i_level](torch.cat([h, adapter_in[i_level]], dim=1))
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.adapter(h)  # Apply Adapter
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

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
    model = SAFMN_adenc4(dim=48, n_blocks=8, ffn_scale=2.0, upscaling_factor=16,
                         vae_weight='/data/liuyidi/nitre_2023_dehaze/data_dehaze/uhdformer/experiments/KL_16_restormer/models/net_g_latest.pth',
                         config='/data/liuyidi/nitre_2023_dehaze/data_dehaze/uhdformer/experiments/KL_16_restormer/kl_16.yml').to('cuda')
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