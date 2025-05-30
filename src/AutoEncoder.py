from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import reshape


class ShapeNavigator(nn.Module):
    def __init__(self, label=""):
        super().__init__()
        self.label = label

    def forward(self, x):
        print(f"[{self.label}] shape: {x.shape}")
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, ffn_hidden=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.transpose(1, 2)  # (B, T, C) -> (B, C, T)


class AutoEncoder(nn.Module):
    def __init__(self, n_mel: int = 80, latent_size: int = 128, num_heads: int = 2):
        super(AutoEncoder, self).__init__()
        self.n_mel = n_mel
        self.latent_size = latent_size
        self.feature_len = None  # set after first forward

        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(n_mel, n_mel // 2, kernel_size=5, stride=1)),
            ("relu1", nn.ReLU()),
            ("attn", SelfAttention(embed_dim=n_mel // 2, num_heads=num_heads)),
            ("relu2", nn.ReLU()),
            ("conv2", nn.Conv1d(n_mel // 2, n_mel // 4, kernel_size=7, stride=2)),
            ("relu3", nn.ReLU()),
            ("conv3", nn.Conv1d(n_mel // 4, n_mel // 4, kernel_size=11, stride=3)),
            ("relu4", nn.ReLU()),
        ]))
        self.encoder_flatten = nn.Flatten()
        self.encoder_projection: nn.Module | None = None  # to be initialized on forward
        self.decoder_input_linear: nn.Module | None = None  # to be initialized on forward
        self.decoder_unflatten: nn.Module | None = None  # to be initialized on forward
        self.decoder_post = nn.Sequential(OrderedDict([
            ("deconv1", nn.ConvTranspose1d(n_mel // 4, n_mel // 4, kernel_size=11, stride=3)),
            ("relu1", nn.ReLU()),
            ("deconv2", nn.ConvTranspose1d(n_mel // 4, n_mel // 2, kernel_size=7, stride=2)),
            ("relu2", nn.ReLU()),
            ("deconv3", nn.ConvTranspose1d(n_mel // 2, n_mel, kernel_size=5, stride=1)),
        ]))


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        reshape_flag = False
        batch_size, n_mic, h, w = None, None, None, None
        if x.dim() > 3:
            batch_size, n_mic, h, w = x.shape
            x = x.reshape(batch_size * n_mic, h, w)
            reshape_flag = True
        ret = self.encoder(x)
        if self.encoder_projection is None:
            flattened_shape = self.encoder_flatten(ret).shape[1]
            self.encoder_projection = nn.Linear(flattened_shape, self.latent_size).to(ret.device)
            self.decoder_input_linear = nn.Linear(self.latent_size, flattened_shape).to(ret.device)
            self.decoder_unflatten = nn.Unflatten(1, unflattened_size=ret.shape[1:])
        ret = self.encoder_flatten(ret)
        ret = self.encoder_projection(ret)
        if reshape_flag:
            ret = ret.reshape(batch_size, n_mic, self.latent_size)
        return ret

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.decoder_input_linear is not None and self.decoder_unflatten is not None, "Uninitializd layer"
        z = self.decoder_input_linear(z)
        z = self.decoder_unflatten(z)
        x = self.decoder_post(z)
        return x

    def forward(self, x, require_latent=False):
        if x.dim() > 3:
            batch_size, _1, h, w = x.shape
            x = x.reshape(batch_size, h, w)
        z = self.encode(x)
        x_hat = self.decode(z)
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)

        if require_latent:
            return z, x_hat
        return x_hat

    @staticmethod
    def from_structure_hyper_and_checkpoint(hyper: dict, checkpoint_path: str, device):
        model = AutoEncoder.from_config(hyper, device)

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    @staticmethod
    def from_config(hyper: dict, device):
        n_mel = hyper.get('n_mel')
        latent_size = hyper.get('latent_size')
        num_heads = hyper.get('num_heads')
        model = AutoEncoder(n_mel=n_mel, latent_size=latent_size, num_heads=num_heads).to(device)
        if "AutoEncoderInitDummyInput" in hyper:
            shape = hyper["AutoEncoderInitDummyInput"]["shape"]
            dummy_input = torch.randn(*shape, device=device)
            with torch.no_grad():
                model.encode(dummy_input)  # 激活lazy层
        return model
        
    def get_encoder(self, require_flatten=True, require_projection=True):
        if require_projection and self.encoder_projection is None: 
           raise RuntimeError("AutoEncode are not initialized yet; please call forward() at least once")
        layers = OrderedDict()
        layers["encoder"] = self.encoder

        if require_flatten:
            layers["flatten"] = self.encoder_flatten

        if require_projection:
            assert self.encoder_projection is not None, "encoder_projection is not initialized. Please run a forward pass first."
            layers["projection"] = self.encoder_projection

        return nn.Sequential(layers)

