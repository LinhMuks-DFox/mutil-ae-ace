from typing import Union

import torch
import torchaudio
import yaml

from src.AutoEncoder import AutoEncoder
from src.LatentDataset import RIRWaveformToMelTransform, LatentTransform
from . import hyperparameters as hyp
from . import options as opt
from src.LightPropagationAndCameraResponse import LightPropagation, CameraResponse

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

        self.light_propagate = LightPropagation(hyp.Distance, hyp.bias, hyp.std)
        self.camera = CameraResponse(hyp.SignalSourceSampleRate, hyp.CameraFrameRate)
        self.adjust = AdjustForResNet()

    @staticmethod
    def blinky_data_normalize(data: torch.Tensor):
        with torch.no_grad():
            data_min = torch.min(data)
            de_min = data - data_min
            return de_min / torch.max(de_min)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        ret = self.to_latent(x)
        ret = self.blinky_data_normalize(ret)
        ret = self.light_propagate(ret)
        ret = self.camera(ret)
        ret = self.adjust(ret)
        return ret


if __name__ == "__main__":
    dummy_data = torch.rand(1, 5, 300)  # 5s * 4LED * 15 nyquist
    light = LightPropagation(
        distance=5, bias=0.1, std=0.05
    )
    camera = CameraResponse(
        signal_source_sample_rate=15, frame_rate=30,
    )
    adjust = AdjustForResNet()
    dummy_data = light(dummy_data)
    print("before camera:", dummy_data.shape)
    dummy_data = camera(dummy_data)
    print("After camera", dummy_data.shape)
    dummy_data = adjust(dummy_data)
    print(dummy_data.shape)
