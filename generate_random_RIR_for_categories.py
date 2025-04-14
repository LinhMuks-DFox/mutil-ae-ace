import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import torch


def main():
    epilog = """\
输出说明：
  每个类别 (ESC-50 中的 0~49) 会对应一个 RIR 文件：

    rir_cat_{类别编号}.pt

  每个 RIR 文件是一个 tensor，形状为 [num_mics, max_rir_length]，每一行表示一个麦克风通道下的 RIR impulse response。
  所有 RIR 都被统一为相同长度，短的会进行 zero-padding，长的则保留。

示例调用：
  python generate_random_RIR_for_categories.py \\
      --num_mics 4 \\
      --categories 0 1 2 3 \\
      --out_dir rir_outputs \\
      --plot
"""

    parser = argparse.ArgumentParser(
        description="Generate filters (RIR) for ESC-50 classes.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num_mics', type=int, default=4,
                        help='Number of microphones to place in the room')
    parser.add_argument('--categories', nargs='+', default=list(range(50)),
                        help='List of category indices (0-49) from ESC-50 to process')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Output directory to save the RIR tensors')
    parser.add_argument('--plot', action='store_true', help='Show a 2D plot of the room, source, and mic positions')
    args = parser.parse_args()

    # 固定房间大小
    room_dim = [8.0, 6.0, 4.0]  # 改为3D房间，增加高度维度
    fs = 44100  # 采样率

    # 为 50 个类别预先定义/随机生成声源位置
    # (这里只演示随机生成，可按需改成固定映射或其他逻辑)
    np.random.seed(42)
    source_positions = {}
    for cat in range(50):
        sx = np.random.uniform(0.5, room_dim[0] - 0.5)
        sy = np.random.uniform(0.5, room_dim[1] - 0.5)
        sz = np.random.uniform(0.5, room_dim[2] - 0.5)
        source_positions[cat] = [sx, sy, sz]

    # 先预先存储所有 RIR，便于计算最大长度
    all_rir_data = {}  # 结构: { category: [rir_mic0, rir_mic1, ... rir_micN], ... }

    for cat in args.categories:
        # 创建房间
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            absorption=0.4,
            max_order=10
        )

        # 添加声源
        source_loc = source_positions[cat]
        room.add_source(source_loc)

        # 随机摆放麦克风
        mic_positions = []
        for i in range(args.num_mics):
            mx = np.random.uniform(0.3, room_dim[0] - 0.3)
            my = np.random.uniform(0.3, room_dim[1] - 0.3)
            mz = np.random.uniform(0.3, room_dim[2] - 0.3)
            mic_positions.append([mx, my, mz])

        # Plot the layout if the plot flag is set
        if args.plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(0, room_dim[0])
            ax.set_ylim(0, room_dim[1])
            ax.set_zlim(0, room_dim[2])

            # 绘制声源位置
            ax.scatter(source_loc[0], source_loc[1], source_loc[2], c='red', marker='*', s=100, label='Source')

            # 绘制麦克风位置
            for i, (mx, my, mz) in enumerate(mic_positions):
                ax.scatter(mx, my, mz, c='blue', marker='o')
                ax.text(mx + 0.05, my + 0.05, mz, f'Mic{i}', fontsize=9)

            ax.set_title(f"Category={cat}")
            plt.savefig(f"{args.out_dir}/categroy{cat}_plot.png")

        # 将麦克风阵列添加到房间
        mic_array = np.array(mic_positions).T  # shape = (3, num_mics)
        room.add_microphone_array(mic_array)

        # 计算 RIR
        room.compute_rir()

        # 收集该类别下，每个麦克风的 RIR
        rir_for_category = [room.rir[mic_i][0] for mic_i in range(args.num_mics)]
        all_rir_data[cat] = rir_for_category

    # ------------------------------
    # 统一 RIR 长度 (零填充) 并保存
    # ------------------------------

    # 1) 找到所有 RIR 的最大长度
    max_len = 0
    for cat in args.categories:
        for rir in all_rir_data[cat]:
            if len(rir) > max_len:
                max_len = len(rir)
    print(f"Max RIR length across all categories/mics: {max_len}")

    # 2) 转换为固定长度的张量
    #    若你希望截断过长的 RIR，可在这里调整逻辑
    for cat in args.categories:
        rir_list = []
        for mic_i, rir in enumerate(all_rir_data[cat]):
            rir_array = np.array(rir, dtype=np.float32)
            curr_len = len(rir_array)

            if curr_len < max_len:
                # 末尾零填充
                padded = np.pad(rir_array, (0, max_len - curr_len), mode='constant', constant_values=0)
            else:
                # 如果长度超出 max_len，可以视需求选择截断，也可以保留
                # 这里示例中直接保留，不截断
                padded = rir_array

            rir_list.append(padded)

        # rir_list 是一个 [num_mics, max_len] 的 2D 数组
        rir_tensor = torch.tensor(rir_list)  # shape: (num_mics, max_len)

        # 将结果保存到文件
        out_file = f"{args.out_dir}/rir_cat_{cat}.pt"
        torch.save(rir_tensor, out_file)
        print(f"Saved RIR for category {cat} to {out_file}")


if __name__ == "__main__":
    main()
