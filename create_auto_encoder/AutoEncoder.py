import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """Audio AutoEncoder using 2D CNNs and a Multihead Attention layer"""

    def __init__(self, latent_size: int, num_heads: int = 2):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, H/4, W/4]
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=2)  # [B, C, H, W] -> [B, C, HW]
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, H, W]
            nn.Sigmoid(),
        )
    
    def encode(spectrogram: torch.Tensor) -> torch.Tensor:
        pass
    
    
    def decode(latent: torch.Tensor) -> torch.Tensor:
        pass
    