import torch
# import pytorch_lightning as pl # type: ignore
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
import sys
sys.path.append("/code/UHDformer-main")

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from basicsr.archs.encoder_2_2 import Encoder, Decoder
from basicsr.utils.distributions.distributions import DiagonalGaussianDistribution

from basicsr.utils.vae_util import instantiate_from_config



   


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
        h,adapters = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()
        return z,adapters

    def decode(self, z,adapters):
        z = self.post_quant_conv(z)
        dec = self.decoder(z,adapters)
        return dec

    def forward(self, input, sample_posterior=True):
        #采样过程合并到encode中
        z,posterior,adapters = self.encode(input)
        dec = self.decode(z,adapters)
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


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x




#测试模型
if __name__ == '__main__':
    import yaml
    with open("/code/UHDformer-main/options/adapter/kl_f16.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = instantiate_from_config(config["model"]).cuda()
    #计算模型中可训练参数量
    para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"model parameters number:{para_num}")    
    # from thop import profile
    # from thop import clever_format
    input = torch.randn(1,3,256,256).cuda()
    # flops,params = profile(model,inputs=(input,))
    # flops,params = clever_format([flops,params], "%.3f")
    # print(f"params:{params},flops:{flops}")
    output = model(input)
    



