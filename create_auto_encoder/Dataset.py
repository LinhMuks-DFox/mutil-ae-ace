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
    def __len__(self):
        return len(self.underlying_data_reader)
    
    @staticmethod
    def from_yaml(yaml_obj, train=True, eval=True):
        """
        Create AudioSet dataset(s) from YAML configuration.
        
        Args:
            yaml_obj: The loaded YAML configuration object.
            train: Whether to load the training dataset.
            eval: Whether to load the evaluation dataset.
        
        Returns:
            A tuple (train_dataset, eval_dataset), where each may be None.
        """
        train_dataset = None
        eval_dataset = None

        if train and "train" in yaml_obj["Dataset"]["Audioset"]:
            t_cfg = yaml_obj["Dataset"]["Audioset"]["train"]
            train_dataset = AudioSet(
                t_cfg["json_path"],
                t_cfg["sample_path"],
                None
            )

        if eval and "eval" in yaml_obj["Dataset"]["Audioset"]:
            e_cfg = yaml_obj["Dataset"]["Audioset"]["eval"]
            eval_dataset = AudioSet(
                e_cfg["json_path"],
                e_cfg["sample_path"],
                None
            )

        return train_dataset, eval_dataset
  
    
  
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