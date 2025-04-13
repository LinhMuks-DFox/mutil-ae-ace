import torch
import yaml

from src.LatentDataset import RIRWaveformToMelTransform, LatentTransform
from . import options as opt
from src.AutoEncoder import AutoEncoder


class AdjustForResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() >= 3:
            batch, n_mic, nyquis_times_5_times_4led = x.shape
            x = x.view(batch, 1, n_mic * 4, nyquis_times_5_times_4led // 4).contiguous()
        elif x.dim() == 2:
            n_mic, nyquis_times_5_times_4led = x.shape
            x = x.view(1, n_mic * 4, nyquis_times_5_times_4led // 4).contiguous()
        return x


class DataPreprocessor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        with open(opt.AutoEncoderHyper, "r") as f:
            hyper = yaml.safe_load(f)
            self.auto_encoder = AutoEncoder.from_structure_hyper_and_checkpoint(
                hyper["AutoEncoder"], opt.AutoEncoderCheckPoint.resolve(), opt.Device
            )
            self.to_mel = RIRWaveformToMelTransform.from_hyper(hyper, opt.Device)
            self.to_latent = LatentTransform(
                self.to_mel, self.auto_encoder
            )
        self.adjust = AdjustForResNet()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        ret = self.to_latent(x)
        ret = self.adjust(ret)
        return ret
