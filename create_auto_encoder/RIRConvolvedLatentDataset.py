import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

# 假设你的 AutoEncoder 在 create_auto_encoder.AutoEncoder 里
# 实际中请根据你的项目结构来导入
from create_auto_encoder.AutoEncoder import AutoEncoder

class RIRConvolvedLatentDataset(Dataset):
    """
    并行处理多通道RIR卷积数据，使用预训练AutoEncoder提取latent。
    - 加载 .pt 文件得到 (data_tensor, label_tensor)，其中 data_tensor.shape=[N, n_mics, wave_len]
    - 取出单条样本 -> shape=[n_mics, wave_len]
    - 视作批量大小=n_mics，一次性做Log-Mel+to_db -> [n_mics, n_mels, time]
    - 用AutoEncoder.encode并行得到 [n_mics, latent_dim]
    - 返回 (latent, label)
    """

    def __init__(
        self,
        rir_convolved_pt: str,      # 指向含有 (data_tensor, label_tensor) 的 .pt文件
        split: str,                 # "train" / "validate" / "test"
        autoencoder_ckpt: str,      # 预训练好的AutoEncoder权重文件
        device: str = "cuda",
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        center: bool = False
    ):
        super().__init__()

        # === 1) 加载 (data_tensor, label_tensor) ===
        dataset_dict = torch.load(rir_convolved_pt, map_location=device)
        if split not in dataset_dict:
            raise ValueError(f"split='{split}' 不存在于 {rir_convolved_pt}，可用的split包括: {list(dataset_dict.keys())}.")
        self.data_tensor, self.label_tensor = dataset_dict[split]  # data: [N, n_mics, wave_len], label: [N]

        # === 2) 加载 AutoEncoder，并只需其中 state_dict ===
        ckpt = torch.load(autoencoder_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt  # 如果只保存了model本身
            
        self.autoencoder = AutoEncoder(n_mel=n_mels)  # 你的初始化方式
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.to(device)
        self.autoencoder.eval()  # 推理模式

        # === 3) 构建 Log-Mel + to_db 变换，和之前训练AutoEncoder时保持一致 ===
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center
        ).to(device)

        self.to_db = T.AmplitudeToDB(top_db=80).to(device)

        self.device = device

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, index):
        """
        单条记录:
          wave_3d = [n_mics, wave_len]
          label    = int (或其他格式)
        把 wave_3d 视作 batch=n_mics => [n_mics, 1, wave_len]
        并行计算 Log-Mel => [n_mics, n_mels, time] => encode => [n_mics, latent_dim]
        """
        wave_3d = self.data_tensor[index].to(self.device)   # [n_mics, wave_len]
        label   = self.label_tensor[index].item()           # int label

        # (batch=n_mics, channel=1, time=wave_len)
        wave_3d = wave_3d.unsqueeze(1)  # => [n_mics, 1, wave_len]

        # 并行做 MelSpectrogram 和 to_db
        mel = self.melspec(wave_3d)    # => [n_mics, n_mels, time]
        mel_db = self.to_db(mel)       # => [n_mics, n_mels, time]

        # 并行送入AutoEncoder的 encode => [n_mics, latent_dim]
        latents = self.autoencoder.encode(mel_db)

        return latents, label

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # Create an instance of RIRConvolvedLatentDataset
    dataset = RIRConvolvedLatentDataset(
        rir_convolved_pt="esc50_rir_train.pt",  # or your actual PT file
        split="train",
        autoencoder_ckpt="autoencoder_final.pt",  # or your actual AE checkpoint
        device="cuda",
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        center=False
    )

    print("Dataset size:", len(dataset))

    # Try one sample
    latents, label = dataset[0]
    print("Single sample latents shape:", latents.shape)
    print("Single sample label:", label)

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for latents_batch, labels_batch in loader:
        print("latents_batch shape:", latents_batch.shape) # [4, n_mics, latent_dim]
        print("labels_batch shape:", labels_batch.shape)
        break
