import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset
from lib.AudioSet.transform import TimeSequenceLengthFixer
from AutoEncoder import AutoEncoder

class RIRWaveformToMelTransform(nn.Module):
    """
    从 config 中提取Resample、TimeFixer、MelSpectrogram、AmplitudeToDB等参数，
    并将 wave_3d = [n_mics, wave_len] 转为 [n_mics, n_mels, time]。
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
        
        # 构建 Resample + TimeFixer
        self.resample = T.Resample(orig_freq, new_freq).to(device)
        self.time_fixer = TimeSequenceLengthFixer(new_freq, timefix_length, timefix_mode).to(device)

        # 构建 MelSpectrogram + to_db
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
        输出: [n_mics, n_mels, time]
        """
        # 并行处理
        wave_3d = self.resample(wave_3d)                # => [n_mics, wave_len']
        wave_3d = self.time_fixer(wave_3d)              # => [n_mics, wave_len_fixed]
        wave_3d = wave_3d.unsqueeze(1)                  # => [n_mics, 1, wave_len_fixed]
        mel = self.melspec(wave_3d)                     # => [n_mics, n_mels, time]
        mel_db = self.to_db(mel)                        # => [n_mics, n_mels, time]
        return mel_db


class RIRConvolvedLatentDataset(Dataset):
    """
    读取 RIR 卷积后的数据，并用预训练AE encode => latent。
    - data_tensor: [N, n_mics, wave_len]
    - label_tensor: [N]
    - transform: RIRWaveformToMelTransform
    - autoencoder: 预训练AutoEncoder
    返回 (latents=[n_mics, latent_dim], label)
    """

    def __init__(
        self,
        rir_convolved_pt: str,      # 这个pt文件包含 {split: (data_tensor, label_tensor)}
        split: str,
        transform: nn.Module,
        autoencoder: nn.Module,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        
        # 加载 (data_tensor, label_tensor)
        dataset_dict = torch.load(rir_convolved_pt, map_location=device)
        if split not in dataset_dict:
            raise ValueError(f"split='{split}' 不存在于 {rir_convolved_pt}, 可用={list(dataset_dict.keys())}")
        self.data_tensor, self.label_tensor = dataset_dict[split]
        
        self.transform = transform      # wave -> mel
        self.autoencoder = autoencoder  # encode mel

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        wave_3d = self.data_tensor[idx].to(self.device)  # => [n_mics, wave_len]
        label = self.label_tensor[idx].item()            # => int

        mel_db = self.transform(wave_3d)                 # => [n_mics, n_mels, time]
        latents = self.autoencoder.encode(mel_db)        # => [n_mics, latent_dim]
        return latents, label


def build_rir_convolved_latent_dataset(
    config: dict,
    rir_convolved_pt: str,
    split: str,
    autoencoder_ckpt: str,
    device: str = "cuda"
) -> RIRConvolvedLatentDataset:
    # ======== 1) 构造wave->mel transform ========
    resample_cfg = config["Resample"]
    timefix_cfg  = config["TimeSequenceLengthFixer"]
    mel_cfg      = config["ToLogMelSpectrogram"]

    transform = RIRWaveformToMelTransform(
        orig_freq       = resample_cfg["orig_freq"],
        new_freq        = resample_cfg["new_freq"],
        timefix_mode    = timefix_cfg["mode"],
        timefix_length  = timefix_cfg["fixed_length"],
        mel_sample_rate = mel_cfg["sample_rate"],
        n_fft           = mel_cfg["n_fft"],
        hop_length      = mel_cfg["hop_length"],
        n_mels          = mel_cfg["n_mels"],
        device          = device
    )

    # ======== 2) 构造 AutoEncoder + Lazy 初始化 + load_state_dict ========
    ae_cfg = config["AutoEncoder"]
    autoencoder = AutoEncoder(
        n_mel       = ae_cfg["n_mel"],
        latent_size = ae_cfg["latent_size"],
        num_heads   = ae_cfg["num_heads"]
    ).to(device)

    # 2.1) 若存在 AutoEncoderInitDummyInput，就 dummy forward 一次
    if "AutoEncoderInitDummyInput" in config:
        shape = config["AutoEncoderInitDummyInput"]["shape"]  # e.g. [1,80,501]
        dummy_input = torch.randn(*shape, device=device)
        _ = autoencoder.encode(dummy_input)  # 激活 Lazy Layer

    # 2.2) 加载checkpoint
    ckpt = torch.load(autoencoder_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    autoencoder.load_state_dict(state_dict, strict=False)
    autoencoder.eval()

    # ======== 3) 构造 RIRConvolvedLatentDataset ========
    dataset = RIRConvolvedLatentDataset(
        rir_convolved_pt=rir_convolved_pt,
        split=split,
        transform=transform,
        autoencoder=autoencoder,
        device=device
    )
    return dataset

if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader

    with open("configs/hyperpara.yml", "r") as f:
        config_all = yaml.safe_load(f)

    dataset = build_rir_convolved_latent_dataset(
        config=config_all,
        rir_convolved_pt="5mic_esc50_rir_train.pt",
        split="train",
        autoencoder_ckpt="AutoEncoder.pt",
        device="cuda"
    )

    print("Dataset len =", len(dataset))
    latents, label = dataset[0]
    print("latents.shape =", latents.shape, "label =", label)

    loader = DataLoader(dataset, batch_size=4)
    for latents_batch, labels_batch in loader:
        print("Batch latents:", latents_batch.shape, "Batch labels:", labels_batch.shape)
        break