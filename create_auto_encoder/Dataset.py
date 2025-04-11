from lib.AudioSet.IO import JsonBasedAudioSet
import torch.utils.data as tch_data
import typing
import torch
class AudioSet(tch_data.Dataset): 
    def __init__(self, json_path, audio_path, transform: typing.Callable[[torch.Tensor], torch.Tensor] | None):
        self.underlying_data_reader = JsonBasedAudioSet(json_path, audio_path)
        self.transform = transform if transform else None
    def __getitem__(self, idx: int):
        sample = self.underlying_data_reader[idx][0]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
if __name__ == "__main__":
    import yaml
    with open("other_configs.yml", "r") as f:
        cfg = yaml.safe_load(f)
    
    train_set = AudioSet(
        cfg["Dataset"]["Audioset"]["train"]["json_path"],
        cfg["Dataset"]["Audioset"]["train"]["sample_path"],
        None
    )
    
    print(train_set[0])