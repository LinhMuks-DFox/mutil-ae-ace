import torch
import torchaudio
from typing import Union
from . import hyperparameters as hyp
from lib.AudioSet.transform import TimeSequenceLengthFixer
from src.LightPropagationAndCameraResponse import CameraResponse, LightPropagation

class AdjustForResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() >= 3:
            batch, n_mic, t = x.shape
            x = x.reshape(batch, 1, n_mic * 4, t // 4).contiguous()
        elif x.dim() == 2:
            n_mic, nyquis_times_5_times_4led = x.shape
            x = x.reshape(1, n_mic * 4, nyquis_times_5_times_4led // 4).contiguous()
        return x


class ToSoundPower(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x ** 2
        return x


class DataPreprocessor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.time_fix = TimeSequenceLengthFixer(hyp.AudioDuration, hyp.ResampleTo, "s")
        self.down_sample = torchaudio.transforms.Resample(orig_freq=hyp.AudioSampleRate, new_freq=hyp.ResampleTo)
        self.sound_power = ToSoundPower()
        self.light_propagate = LightPropagation(hyp.Distance, hyp.bias, hyp.std)
        self.camera = CameraResponse(hyp.SignalSourceSampleRate, hyp.CameraFrameRate)
        self.adjust = AdjustForResNet()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        ret = self.down_sample(x)
        ret = self.time_fix(ret)
        ret = self.sound_power(ret)
        ret = self.light_propagate(ret)
        ret = self.camera(ret)
        # print(ret.shape)
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
