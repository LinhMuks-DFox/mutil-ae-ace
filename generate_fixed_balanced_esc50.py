import os
import torch
import yaml
from pathlib import Path
from lib.esc50_io.ESC50IO.ESC50 import ESC50, label_balanced_split_esc50

def dataset_to_tensor(subset, num_classes):
    data_list = []
    label_list = []
    for i in range(len(subset)):
        audio, label_idx = subset[i]
        data_list.append(audio)

        # 创建一个 num_classes 维度的独热向量
        one_hot_label = torch.zeros(num_classes, dtype=torch.float32)
        one_hot_label[label_idx] = 1.0
        label_list.append(one_hot_label)

    # 堆叠所有 audio，假设每个音频 Tensor 的形状一致
    data_tensor = torch.stack(data_list, dim=0)
    label_tensor = torch.stack(label_list, dim=0)
    return data_tensor, label_tensor

def main():
    print("=== Starting process... ===")

    # 读取数据集配置信息
    print("Loading dataset configuration from 'dataset_info.yml' ...")
    with open("configs/dataset_info.yml", "r") as f:
        dataset_info = yaml.safe_load(f)

    # 获取 ESC50 数据存放路径
    esc50_path = dataset_info["Dataset"]["ESC50"]["path"]
    print(f"ESC50 dataset path: {esc50_path}")

    # 创建 ESC50 数据集实例
    print("Creating ESC50 dataset instance (subset='all')...")
    dataset = ESC50(data_dir=esc50_path, download=False, subset="all")

    # 按 0.8, 0.1, 0.1 分别平衡分割为 train, test, validate
    print("Splitting dataset into train/test/validate (0.8 / 0.1 / 0.1)...")
    train_set, test_set, val_set = label_balanced_split_esc50(dataset, [0.8, 0.1, 0.1])

    # 获取类目总数
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")

    # 将数据集转换为 (data_tensor, label_tensor) 并将 label 转为 one-hot
    print("Converting subsets to (data_tensor, label_tensor) with one-hot labels...")
    train_data, train_labels = dataset_to_tensor(train_set, num_classes)
    test_data, test_labels = dataset_to_tensor(test_set, num_classes)
    val_data, val_labels = dataset_to_tensor(val_set, num_classes)

    # 打包并保存
    output_path = "fixed_rir_esc50/esc50_fixed_dataset.pt"
    print(f"Saving fixed dataset to {output_path} ...")
    fixed_dataset = {
        "train": (train_data, train_labels),
        "test": (test_data, test_labels),
        "validate": (val_data, val_labels)
    }
    torch.save(fixed_dataset, output_path)
    print("=== Done! Fixed dataset saved. ===")

if __name__ == "__main__":
    main()