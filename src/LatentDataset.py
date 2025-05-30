import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset
from lib.AudioSet.transform import TimeSequenceLengthFixer

from src.AutoEncoder import AutoEncoder


########################################
# 1) 仅负责“读取 data_tensor”的简单 Dataset
########################################
class DataTensorDataset(Dataset):
    """
    只负责从 data_tensor / label_tensor 中读取 raw wave，
    不做任何 transform。允许后续在外部套上 BC 或其他操作。
    """

    def __init__(self, data_tensor: torch.Tensor, label_tensor: torch.Tensor, device="cpu", n_cls=50):
        """
        Args:
            data_tensor: shape=[N, n_mics, wave_len]
            label_tensor: shape=[N] or [N, ...]，取决于你的标签格式
        """
        super().__init__()
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.device = device  # kept for backward compatibility, not used
        self.n_cls = n_cls

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        wave_3d = self.data_tensor[idx]  # => [n_mics, wave_len]
        label = self.label_tensor[idx]  # 可能是 int，也可能是 Tensor
        if isinstance(label, int) or label.ndim == 0:
            label = torch.eye(self.n_cls)[label]
        return wave_3d, label

    @staticmethod
    def from_datatensor_path(split: str, data_tensor_path: str, device="cpu", n_cls=50):
        """
            split: train test validate in string
        """
        dataset_dict = torch.load(data_tensor_path, map_location="cpu", weights_only=True)
        if split not in dataset_dict:
            raise ValueError(f"split='{split}' not found in {data_tensor_path}, available={list(dataset_dict.keys())}")

        data_tensor, label_tensor = dataset_dict[split]
        return DataTensorDataset(data_tensor, label_tensor, device)


########################################
# 2) 将“raw wave -> mel -> AE latent”封装成可复用的 nn.Module
########################################
class RIRWaveformToMelTransform(nn.Module):
    """
    假设你的超参中有:
    config["Resample"], config["TimeSequenceLengthFixer"], config["ToLogMelSpectrogram.py"]
    这里就和之前的 WaveformToMelTransform 类似。
    """

    def __init__(
            self,
            orig_freq: int,
            new_freq: int,
            timefix_mode: str,
            timefix_length: int,
            mel_sample_rate: int,
            n_fft: int,
            hop_length: int,
            n_mels: int,
            device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        # 1) Resample
        self.resample = T.Resample(orig_freq, new_freq).to(device)
        # 2) Time Fix
        self.time_fixer = TimeSequenceLengthFixer(new_freq, timefix_length, timefix_mode).to(device)
        # 3) Mel + AmplitudeToDB
        self.melspec = T.MelSpectrogram(
            sample_rate=mel_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(device)
        self.to_db = T.AmplitudeToDB().to(device)

    def forward(self, wave_3d: torch.Tensor) -> torch.Tensor:
        """
        wave_3d: [n_mics, wave_len]
        返回: [n_mics, n_mels, time]
        """
        wave_3d = self.resample(wave_3d)
        wave_3d = self.time_fixer(wave_3d)
        wave_3d = wave_3d.unsqueeze(1)  # => [n_mics, 1, wave_len_fixed]
        mel = self.melspec(wave_3d)  # => [n_mics, n_mels, time]
        mel_db = self.to_db(mel)
        return mel_db

    @staticmethod
    def from_hyper(hyper: dict, device):
        resample_cfg = hyper["Resample"]
        timefix_cfg = hyper["TimeSequenceLengthFixer"]
        mel_cfg = hyper["ToLogMelSpectrogram"]
        mel_transform = RIRWaveformToMelTransform(
            orig_freq=resample_cfg["orig_freq"],
            new_freq=resample_cfg["new_freq"],
            timefix_mode=timefix_cfg["mode"],
            timefix_length=timefix_cfg["fixed_length"],
            mel_sample_rate=mel_cfg["sample_rate"],
            n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"],
            device=device
        )
        return mel_transform


########################################
# 3) 封装“wave -> mel -> AutoEncoder latent” 的 transform
########################################
class LatentTransform(nn.Module):
    """
    负责：raw wave -> mel -> AE encode -> latent
    和 RIRWaveformToMelTransform 配合使用
    """

    def __init__(self, mel_transform: nn.Module, autoencoder: nn.Module):
        """
        Args:
            mel_transform: 上面定义的 RIRWaveformToMelTransform (或类似)
            autoencoder: 预训练的 AutoEncoder
        """
        super().__init__()
        self.mel_transform = mel_transform
        self.autoencoder = autoencoder

    def forward(self, wave_3d: torch.Tensor):
        """
        wave_3d: [n_mics, wave_len]
        return: latents => [n_mics, latent_dim]
        """
        mel_db = self.mel_transform(wave_3d)  # => [n_mics, n_mels, time]
        latents = self.autoencoder.encode(mel_db)  # => [n_mics, latent_dim]
        return latents


########################################
# 4) Dataset包装：把“DataTensorDataset” + “LatentTransform”组合起来
########################################
class LatentTransformDataset(Dataset):
    """
    外部可以先做BC，或者把“DataTensorDataset”包在BC外，最后再进本Dataset。
    """

    def __init__(self, base_dataset: Dataset, latent_transform: nn.Module):
        """
        Args:
            base_dataset: 返回 (wave_3d, label) 的Dataset
            latent_transform: 上面定义的LatentTransform实例
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.latent_transform = latent_transform

    def __len__(self):
        return len(self.base_dataset)

    def change_base_dataset(self, dataset: Dataset):
        self.base_dataset = dataset
        return self

    def __getitem__(self, idx):
        wave_3d, label = self.base_dataset[idx]  # wave_3d => [n_mics, wave_len]
        latents = self.latent_transform(wave_3d)
        return latents, label


########################################
# 5) 构造函数示例：与超参衔接
########################################
def build_rir_latent_dataset(config: dict, data_pt: str, split: str, autoencoder_ckpt: str, device="cuda"):
    """
    类似于原先的build_latent_dataset，只是拆分成两部分。
    data_pt里包含: {split: (data_tensor, label_tensor)}
    """
    # 1) 读取 raw_data
    base_dataset = DataTensorDataset.from_datatensor_path(split, data_pt, device=device)

    # 2) 构造 RIRWaveformToMelTransform
    mel_transform = RIRWaveformToMelTransform.from_hyper(config, device)

    # 3) 构造预训练AutoEncoder
    autoencoder = AutoEncoder.from_structure_hyper_and_checkpoint(config["AutoEncoder"], autoencoder_ckpt, device)

    # 4) 构造 LatentTransform 并组合成Dataset
    latent_transform = LatentTransform(mel_transform, autoencoder)
    dataset = LatentTransformDataset(base_dataset, latent_transform)
    return dataset


if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader

    # 加载超参数
    with open("../configs/auto_encoder_hyperpara.yml", "r") as f:
        config_all = yaml.safe_load(f)

    # 构建 dataset
    dataset = build_rir_latent_dataset(
        config=config_all,
        data_pt="5mic_esc50_rir_train.pt",
        split="train",
        autoencoder_ckpt="autoencoder_checkpoint/run_20250413_155634/checkpoints/epoch_300.pt",
        device="cuda"
    )

    print("Dataset size =", len(dataset))
    latents, label = dataset[0]
    print("latents.shape =", latents.shape, "label =", label)

    # 批量读取
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for latents_batch, labels_batch in loader:
        print("Batch latents:", latents_batch.shape)
        print("Batch labels:", labels_batch.shape)
        break
