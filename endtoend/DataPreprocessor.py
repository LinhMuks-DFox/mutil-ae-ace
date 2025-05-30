import torch
import yaml

from src.LatentDataset import RIRWaveformToMelTransform
from . import options as opt


class AdjustForResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # if x.dim() >= 5:
        #     batch, n_mic, _1, h, w = x.shape
        #     x = x.reshape(batch, n_mic, h, w).contiguous()
        # elif x.dim() == 4:
        if x.dim() == 3:
            n_mic, h, w = x.shape
            return x
        if x.dim() == 4:
            batch_size, n_mic, h, w = x.shape
            x = x.reshape(batch_size, n_mic, h, w).contiguous()

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
        # print(ret.shape)

        ret = self.adjust(ret)
        return ret
