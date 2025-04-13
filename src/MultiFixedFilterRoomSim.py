import os
import torch
import torch.nn as nn
import torchaudio.transforms as T

class MultiFixedFilterRoomSim(nn.Module):
    """
    A PyTorch module that can handle multiple categories' RIRs at once.
    It loads all RIRs from the given directory and applies convolution
    to the input audio based on the provided category id in forward().

    Args:
        rir_dir (str): The directory where rir_cat_*.pt files are located.
                       Expecting files like: rir_cat_0.pt, rir_cat_1.pt, ...
        categories (list or range): Which ESC-50 category IDs to load (0..49).
        mode (str): 'full' or 'valid' or 'same'. Used by torchaudio.transforms.Convolve.

    Forward Input:
        audio (torch.Tensor): shape [n_channel, n_sample].
                              Typically n_channel=1 for mono.
        cat_id (int): the category ID specifying which RIR to use.

    Forward Output:
        torch.Tensor: shape [n_mics, out_len] after RIR-based convolution.
    """

    def __init__(self, rir_dir: str, categories, mode='full'):
        super().__init__()

        self.rir_dir = rir_dir
        self.categories = categories if isinstance(categories, (list, range)) else [categories]
        self.mode = mode
        self.convolver = T.Convolve(mode=self.mode)

        # 1) 依次加载所有需要的类别 RIR
        #    并将每个 RIR 注册成 buffer
        for cat_id in self.categories:
            rir_path = os.path.join(self.rir_dir, f"rir_cat_{cat_id}.pt")
            rir_tensor = torch.load(rir_path)  # shape [n_mics, rir_len]
            # 注册成buffer，命名规则如 "rir_0", "rir_1" ...
            self.register_buffer(f"rir_{cat_id}", rir_tensor)

    def forward(self, audio: torch.Tensor, cat_id: int) -> torch.Tensor:
        """
        Convolve the input audio with the specified category's RIR.

        Args:
            audio (torch.Tensor): shape [n_channel, n_sample].
            cat_id (int): which category's RIR to use.

        Returns:
            torch.Tensor: shape [n_mics, out_len], convolved audio for each microphone.
        """

        # 2) 取出我们在构造函数里注册的 RIR
        #    "rir_0", "rir_1", ... 这样的命名
        buffer_name = f"rir_{cat_id}"
        if not hasattr(self, buffer_name):
            raise ValueError(f"RIR buffer not found for category {cat_id}")

        rir = getattr(self, buffer_name)  # shape [n_mics, rir_len]

        out_list = []
        for mic_idx in range(rir.size(0)):
            # rir[mic_idx]: shape [rir_len]
            kernel = rir[mic_idx].unsqueeze(0)  # shape [1, rir_len]
            # 做卷积
            out_wave = self.convolver(audio, kernel)
            # out_wave: shape [n_channel, out_len] (usually n_channel=1)
            out_list.append(out_wave)

        # 将不同麦克风结果在 "channel" 维度拼起来 => shape [n_mics, out_len]
        return torch.cat(out_list, dim=0)
    
if __name__ == "__main__":
    import torch

    # 假设你已经在 rir_dir 目录下，准备好了
    # rir_cat_0.pt、rir_cat_1.pt 等文件
    rir_dir = "10mic"
    categories_to_load = [0, 1]  # 比如只加载0号和1号类别

    # 实例化模块
    sim = MultiFixedFilterRoomSim(
        rir_dir=rir_dir,
        categories=categories_to_load,
        mode='full'
    )
    sim.eval()  # 如果只在推理时使用，可将其设为eval模式

    # 准备一个简单的测试音频：单声道，1秒钟的随机噪声 (采样率未固定，这里只做演示)
    # audio.shape = [1, 16000]
    audio = torch.randn(1, 16000)

    # 使用类别 ID=0 进行卷积模拟
    cat_id = 0
    out = sim(audio, cat_id)
    print("输出张量形状:", out.shape)
    # out.shape = [n_mics, out_len]
    
    # 如果你想使用类别 ID=1
    cat_id = 1
    out2 = sim(audio, cat_id)
    print("输出张量形状 (cat=1):", out2.shape)