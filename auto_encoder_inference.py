import subprocess
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from ae_ideal.DataPreprocessor import DataPreprocessor
from lib.esc50_io.ESC50IO import get_index_to_category 

def get_git_branch():
    """通过 subprocess 获取当前 git 分支名称"""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8").strip()
        return branch
    except Exception as e:
        print(f"Warning: Could not detect git branch ({e}). Using 'unknown'.")
        return "unknown"

def load_dataset(path):
    """加载并解包数据集"""
    dataset_tensor = torch.load(path)
    # 假设字典顺序为 train, test, validate
    return list(dataset_tensor.values())

def group_data_by_category(data_tuple, category_map):
    """
    将数据按类别分组
    输入: data_tuple = (all_data_tensor, all_labels_tensor)
    输出: dict {"label_name": [tensor_sample, ...]}
    """
    all_data, all_labels = data_tuple
    grouped_data = defaultdict(list)

    for i in range(len(all_data)):
        sample = all_data[i]
        label_vec = all_labels[i]
        
        # 获取类别名称
        idx = label_vec.argmax().item()
        category_name = category_map[idx]
        
        grouped_data[category_name].append(sample)

    return grouped_data

def collect_latents(iterable_data, to_latent_func, device):
    """提取 Latent Code"""
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for label_name, data_list in iterable_data.items():
            for item in data_list:
                # item shape: [1, 220500] -> model input
                tensor_in = item.to(device)
                
                # 获取 latent 并转为 numpy
                latent = to_latent_func(tensor_in) 
                latent_np = latent.squeeze(0).cpu().numpy()
                
                all_latents.append(latent_np)
                all_labels.append(label_name)

    return np.array(all_latents), np.array(all_labels)

def main():
    # 1. 配置环境与路径
    branch_name = get_git_branch()
    print(f"Running on branch: {branch_name}")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    data_path = Path(".") / "data" / "esc50_fixed_dataset.pt"

    # 2. 初始化预处理器 (依赖于当前分支代码)
    preprocessor = DataPreprocessor()
    to_latent = preprocessor.to_latent
    
    # 3. 加载数据
    # 注意：需确保 get_index_to_category 可用，此处假设它是全局或已导入
    # category_map = get_index_to_category() 
    # 临时模拟 map 以保证代码可运行，实际请替换为你的导入
    category_map = {i: f"class_{i}" for i in range(50)} 

    train_set, test_set, val_set = load_dataset(data_path)
    
    # 定义需要处理的数据集 splits
    datasets_to_process = {
        "train": train_set,
        # "test": test_set,   # 如需处理测试集可取消注释
        # "val": val_set
    }

    # 4. 执行处理循环
    for split_name, dataset in datasets_to_process.items():
        print(f"Processing {split_name} set...")
        
        grouped = group_data_by_category(dataset, category_map)
        X, y = collect_latents(grouped, to_latent, device)
        
        # 5. 保存结果 (文件名包含分支信息)
        file_prefix = f"{branch_name}_{split_name}"
        np.save(f"{file_prefix}_X.npy", X)
        np.save(f"{file_prefix}_y.npy", y)
        print(f"Saved: {file_prefix}_X.npy, {file_prefix}_y.npy")

if __name__ == "__main__":
    main()