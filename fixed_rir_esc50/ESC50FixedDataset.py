import torch
from torch.utils.data import Dataset


class ESC50FixedDataset(Dataset):
    """
    读取由 main.py 保存的 esc50_fixed_dataset.pt 文件，
    并作为可直接用于训练/验证/测试的 Dataset。
    
    Args:
        file_path (str): 指向 esc50_fixed_dataset.pt 的路径。
        split (str): "train"、"test" 或 "validate"。
    """

    def __init__(self, file_path: str, split: str = "train"):
        loaded_data = torch.load(file_path)

        if split not in loaded_data:
            raise ValueError(f"Unknown split: '{split}' not found in {file_path}. "
                             f"Expected one of {list(loaded_data.keys())}.")

        self.data_tensor, self.label_tensor = loaded_data[split]

    def __getitem__(self, idx):
        # 返回单条记录 (音频数据, label)
        return self.data_tensor[idx], self.label_tensor[idx]

    def __len__(self):
        # 数据集大小
        return len(self.data_tensor)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 假设之前保存好的固定 ESC50 数据集文件为 "esc50_fixed_dataset.pt"
    dataset_path = "esc50_fixed_dataset.pt"

    # 以 train 分割为例创建数据集实例
    train_dataset = ESC50FixedDataset(dataset_path, split="train")

    # 简单测试：打印数据集大小和前 1 条样本
    print("Number of samples in train split:", len(train_dataset))

    # 创建 DataLoader 并取一个批次演示
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for audio_batch, label_batch in train_loader:
        print("Audio batch shape:", audio_batch.shape)  # (batch_size, waveform_length)
        print("Label batch shape:", label_batch.shape)  # (batch_size, num_classes)
        # 只示例一次就 break
        break
