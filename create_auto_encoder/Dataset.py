import MuxKitAudioSetIO.AudioSet.IO.JsonBasedAudioSet as jba
import torch.utils.data as tch_data
import typing
import torch

class AudioSet(tch_data.Dataset):
    def __init__(self, json_path, audio_path, transform: typing.Callable[[torch.Tensor], torch.Tensor] | None):
        self.underlying_data_reader = jba.JsonBasedAudioSet(json_path, audio_path)
        self.transform = transform if transform else None
    def __get_item__(self, idx:int):
        
        sample = self.underlying_data_reader[idx][0]
        if self.transform:
            sample = self.transform(sample)
        return sample