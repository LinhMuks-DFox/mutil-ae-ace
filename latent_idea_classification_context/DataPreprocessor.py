import torch
import torchaudio.transforms as tch_audioT

from MuxkitTools.audio_tools.transforms import SoundTrackSelector, TimeSequenceLengthFixer
from MuxkitTools.audio_tools.utils import fix_length
from . import hyperparameters as hyp


class DataPreprocessor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.track_selector = SoundTrackSelector(
            hyp.SoundTrack
        )
        self.resampler = tch_audioT.Resample(
            hyp.DataSampleRate, hyp.NewSampleRate
        )
        self.time_fixer = TimeSequenceLengthFixer(
            hyp.AudioDuration, hyp.NewSampleRate
        )

        self.track_selector = SoundTrackSelector(
            hyp.SoundTrack
        )
        self.resampler = tch_audioT.Resample(
            hyp.DataSampleRate, hyp.NewSampleRate
        )
        self.time_fixer = TimeSequenceLengthFixer(
            hyp.AudioDuration, hyp.NewSampleRate
        )

        self.spectrogram_converter = tch_audioT.Spectrogram(
            win_length=2047,
            n_fft=2047
        )
        self.to_db = tch_audioT.AmplitudeToDB()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self.track_selector(x)
        x = self.resampler(x)
        x = self.time_fixer(x)
        # n_mic, n_sample = x.shape
        # x = x.view(n_mic, 1, n_sample).contiguous()
        x = fix_length(x, hyp.NewSampleRate * hyp.AudioDuration)

        x = self.spectrogram_converter(x)
        x = self.to_db(x)
        return x
