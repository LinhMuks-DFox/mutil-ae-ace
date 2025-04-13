import torch
import torchaudio
from typing import Union
from . import hyperparameters as hyp
from lib.AudioSet.transform import TimeSequenceLengthFixer


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


class LightPropagation(torch.nn.Module):
    def __init__(self, distance: float, bias: Union[float, None], std: float):
        super(LightPropagation, self).__init__()
        self.distance = distance
        self.bias = bias
        self.std = std

    def forward(self, x):
        attenuation = 1 / (self.distance ** 2)
        if self.bias is None:
            bias = torch.rand(1).to(x.device)
        else:
            bias = self.bias
        noise = self.std * torch.randn(x.shape).to(x.device)
        x = attenuation * x + bias + noise
        return x


class CameraResponse(torch.nn.Module):
    def __init__(self, signal_source_sample_rate: int,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(CameraResponse, self).__init__()
        self.frame_rate = frame_rate
        self.temperature = temperature
        self.resample = torchaudio.transforms.Resample(
            orig_freq=signal_source_sample_rate,
            new_freq=frame_rate,
            resampling_method='sinc_interp_hann'
        )

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x = self.resample(x)
        # x = deepy.nn.functional.softstaircase(x, self.levels, self.temperature)
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
        self.time_fix = TimeSequenceLengthFixer(hyp.AudioDuration, hyp.AudioSampleRate, "s")
        self.sound_power = ToSoundPower()
        self.light_propagate = LightPropagation(hyp.Distance, hyp.bias, hyp.std)
        self.camera = CameraResponse(hyp.SignalSourceSampleRate, hyp.CameraFrameRate)
        self.adjust = AdjustForResNet()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        ret = self.time_fix(x)
        ret = self.sound_power(ret)
        ret = self.light_propagate(ret)
        ret = self.camera(ret)
        ret = self.adjust(ret)
        print(ret.shape)

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
