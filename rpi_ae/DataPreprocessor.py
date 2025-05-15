import torch
import yaml

from src.AutoEncoder import AutoEncoder
from src.LatentDataset import RIRWaveformToMelTransform, LatentTransform
from . import options as opt
from . import hyperparameters as hyp

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


class RandomRIR(torch.nn.Module):
    pass

class RandomShift(torch.nn.Module):
    pass

class RandomScale(torch.nn.Module):
    pass


def new_random_room():

    room = ...

    yield


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

        self.reset_random_room_interval:int = hyp.ResetRoomInterval()

        self.rand_rir = RandomRIR()
        self.rand_shift = RandomShift()
        self.rand_scale = RandomScale()




    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Given a 1-channel raw-audio in shape of torch.Size([1, 220500], in sample rate of 44100, in a duration of 5-secound perform:
        ---
        0. To a faster approach, may be we can RIRWaveformToMelTransform.resample() first. 44100 -> 16000.
            0.1 RIRWaveformToMelTransform contains resample, time length fixing, and mel spectrogram conversion
            0.2 If the resample is called manually, then the data pipeline needs to be called again manually and in order.

        1. apply random room_rir [(maybe_batch_size), 1, time_axis] -> [(maybe_batch_size), N_MIC, time axis]
        2. randomly shift(cut & zero-padding) **each channel**
            2.1 [(maybe_batch_size), N_MIC, time axis] -> [(maybe_batch_size), N_MIC, time axis]
            2.2 to simulate asyncronized miccrophone, which start record at different time-stemp, but there are 2 situation:
                2.2.1 The microphones were turned on asynchronously, artificially causing a shift in some recordings, 
                      with later microphones only picking up part of the sound event.
                2.2.2 Although the microphone is turned on asynchronously, some has been on for some time, 
                     so some sound events can be captured in full.
            2.3 In order to ensure that the downstream resnet has seen different rooms and is robust to 
                changes in the RIR of the rooms, it is possible to artificially modify the RIR according to the interval

        3. apply random scale, to simulate sound anuetation in air.


        4. Fix the time-length to 5s, by using  RIRWaveformToMelTransform.time_fixer()
        5. Convert to multi-channel audio-data into multi-channel Mel-Spectrogram, by callsing RIRWaveformToMelTransform.melspec() and RIRWaveformToMelTransform.to_db()

        6. Apply Auto-Encoder forward function.

        7. output should be a tensor, which has a shpe of torch.Size[(maybe_batch_size), N_MIC, latent_dim])] 

        """

        pass
