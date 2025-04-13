import torch
from torch.utils.data import DataLoader
import os

# 假设你这两个类和你的工程目录结构如下，如果目录不同请自行调整 import 路径
from src.ESC50FixedDataset import ESC50FixedDataset
from src.MultiFixedFilterRoomSim import MultiFixedFilterRoomSim

def generate_rir_convolved_esc50(
    esc50_fixed_pt,    # 指向原始 esc50_fixed_dataset.pt
    rir_dir,           # 存放 rir_cat_0.pt, rir_cat_1.pt 等文件的目录
    out_dir,           # 输出目录（如 "rir_convolved"）
    split='train',     # "train" / "validate" / "test"
    categories=range(50),  # 如果你只准备了部分类别 RIR，可指定相应的范围
    mode='full',           # MultiFixedFilterRoomSim 的卷积模式
    device='cpu',
    verbose=False          # 新增参数
):
    """
    将 ESC50FixedDataset 的音频全部做一次 RIR 卷积，并把多麦克风结果保存到 out_dir/esc50_rir_{split}.pt。
    最终可得到形状 [N, n_mics, out_len] 的数据张量，以及 [N] 的标签张量。
    """
    # 如果 out_dir 不存在，先创建
    os.makedirs(out_dir, exist_ok=True)
    # 最终的输出文件名
    out_filename = f"esc50_rir_{split}.pt"
    out_pt = os.path.join(out_dir, out_filename)

    # 1) 加载原始 ESC50 数据
    dataset = ESC50FixedDataset(file_path=esc50_fixed_pt, split=split)
    
    # 2) 初始化 RIR 卷积模块
    sim = MultiFixedFilterRoomSim(
        rir_dir=rir_dir,
        categories=categories,  # 比如 range(50) 或者只是一部分分类
        mode=mode
    ).to(device)

    data_list = []
    label_list = []
    
    # 3) 遍历数据，并进行 RIR 卷积
    for i in range(len(dataset)):
        raw_audio, label = dataset[i]  # raw_audio.shape: [waveform_length], label 可能是一维或 one-hot
        # 如果 label 是 one-hot，则需要拿 argmax 变成整数类别
        if isinstance(label, torch.Tensor) and label.dim() == 1 and label.numel() > 1:
            label_id = label.argmax().item()
        else:
            # 可能是 int 或标量 tensor
            label_id = int(label) if isinstance(label, torch.Tensor) else label

        raw_audio = raw_audio.to(device)  # 变成 shape [1, waveform_length]

        if verbose:
            print(f"Processing sample {i}: raw_audio.shape={raw_audio.shape}, label_id={label_id}")

        # 将该样本做 RIR 卷积
        convolved = sim(raw_audio, label_id)  # shape [n_mics, out_len]

        if verbose:
            print(f"Convolved shape={convolved.shape}")

        convolved = convolved.cpu()           # 如需最终存到 CPU 张量里
        
        data_list.append(convolved)
        label_list.append(label_id)

    # 拼接成大张量
    data_tensor = torch.stack(data_list, dim=0)   # [N, n_mics, out_len]
    label_tensor = torch.tensor(label_list, dtype=torch.long)  # [N]

    if verbose:
        print(f"Final data_tensor shape: {data_tensor.shape}, label_tensor shape: {label_tensor.shape}")

    # 4) 以字典形式保存到 pt 文件
    saved_dict = {split: (data_tensor, label_tensor)}
    torch.save(saved_dict, out_pt)

    print(f"【{split}】RIR 卷积后数据: {data_tensor.shape}, 标签: {label_tensor.shape}")
    print(f"已保存到: {out_pt}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run RIR convolution on ESC50 dataset")
    parser.add_argument('--esc50_fixed_pt', type=str, default="esc50_fixed_dataset.pt", help="Path to ESC50 fixed dataset pt file")
    parser.add_argument('--rir_dir', type=str, required=True, help="Directory containing RIR files (e.g. 5mic_rir)")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory. If not provided, defaults to <rir_dir>_rir_convolved")
    parser.add_argument('--mode', type=str, default="full", help="Convolution mode for MultiFixedFilterRoomSim")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--split', type=str, choices=['train', 'validate', 'test', 'all'], default="all", help="Data split to process (train, validate, test, or all)")
    parser.add_argument('--categories', type=str, default="range(50)", help="Categories as a Python expression, e.g., 'range(50)'")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"{args.rir_dir}_rir_convolved"

    try:
        categories = eval(args.categories)
    except Exception as e:
        raise ValueError("Failed to parse categories argument. Please provide a valid Python expression.") from e

    splits = [args.split] if args.split != 'all' else ['train', 'validate', 'test']

    for split in splits:
        generate_rir_convolved_esc50(
            esc50_fixed_pt=args.esc50_fixed_pt,
            rir_dir=args.rir_dir,
            out_dir=args.out_dir,
            split=split,
            categories=categories,
            mode=args.mode,
            device=args.device,
            verbose=args.verbose
        )