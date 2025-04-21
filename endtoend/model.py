from collections import OrderedDict

import torch.nn as nn
from . import options as opt
from torchvision.models import resnet50, resnet18, resnet34
import torch
from . import hyperparameters as hyp
from src.AutoEncoder import AutoEncoder
from src.LightPropagationAndCameraResponse import LightPropagation, CameraResponse
import yaml


class AdjustForResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() >= 3:
            batch, n_mic, nyquis_times_5_times_4led = x.shape
            x = x.reshape(batch, 1, n_mic * 4, nyquis_times_5_times_4led // 4).contiguous()
        elif x.dim() == 2:
            n_mic, nyquis_times_5_times_4led = x.shape
            x = x.reshape(1, n_mic * 4, nyquis_times_5_times_4led // 4).contiguous()
        return x

class EndToEndModel(nn.Module):

    def __init__(self, resnet_type, n_cls):
        super().__init__()
        with open(opt.AutoEncoderHyper, "r") as f:
            hyper = yaml.safe_load(f)
            self.auto_encoder = AutoEncoder.from_config(
                hyper["AutoEncoder"], opt.Device
            )

        self.light_propagation = LightPropagation(hyp.Distance, hyp.bias, hyp.std)
        self.camera_sample = CameraResponse(hyp.SignalSourceSampleRate, hyp.CameraFrameRate)
        self.resnet = get_resnet(resnet_type, n_cls)
        self.adjust = AdjustForResNet()

    def forward(self, x:torch.Tensor):
        batch_size, n_mic, h, w = x.shape
        x = x.reshape(batch_size * n_mic, h, w)
        latent = self.auto_encoder.encode(x)
        latent = latent.reshape(batch_size, n_mic, -1)
        latent = self.adjust(latent)
        latent = self.light_propagation(latent)
        latent = self.camera_sample(latent)
        y = self.resnet(latent)
        return y


def get_resnet(resnet_type, n_cls):
    projection = nn.Conv2d(1, 3, kernel_size=1)
    resnet = {
        18: resnet18,
        34: resnet34,
        50: resnet50
    }[resnet_type](num_classes=n_cls)

    return nn.Sequential(OrderedDict([
        ("Projection", projection),
        ("Resnet", resnet)
    ]))


def make_model(resnet_type, n_cls):
    return EndToEndModel(resnet_type, n_cls)
