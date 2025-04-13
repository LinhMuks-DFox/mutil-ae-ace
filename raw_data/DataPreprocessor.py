from typing import Union

import torch
import torchaudio
import yaml

from src.AutoEncoder import AutoEncoder
from src.LatentDataset import RIRWaveformToMelTransform
from . import hyperparameters as hyp
from . import options as opt


class AdjustForResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() >= 5:
            batch, n_mic, _1, h, w = x.shape
            x = x.view(batch, n_mic, h, w).contiguous()
        elif x.dim() == 4:
            n_mic, _1, h, w = x.shape
            x = x.view(n_mic, h, w).contiguous()
        return x


class DataPreprocessor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        with open(opt.AutoEncoderHyper, "r") as f:
            hyper = yaml.safe_load(f)
        self.to_mel = RIRWaveformToMelTransform.from_hyper(hyper, opt.Device)
        self.adjust = AdjustForResNet()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        ret = self.to_mel(x)
        ret = self.adjust(ret)
        return ret
