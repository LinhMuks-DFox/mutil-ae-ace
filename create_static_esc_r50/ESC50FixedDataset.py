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
        # 加载已经固定好的 ESC50 数据 (train/test/validate)
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