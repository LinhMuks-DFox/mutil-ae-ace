import yaml
from torch.utils.data import DataLoader
from src.LatentDataset import build_rir_latent_dataset
# 加载超参数
with open("configs/auto_encoder_hyperpara.yml", "r") as f:
    config_all = yaml.safe_load(f)

# 构建 dataset
dataset = build_rir_latent_dataset(
    config=config_all,
    data_pt="data/5mic_rir_rir_convolved/esc50_rir_train.pt",
    split="train",
    autoencoder_ckpt="/home/mux/code_workspace/multi-ae-ace/autoencoder_checkpoint/run_20250413_155634/checkpoints/epoch_300.pt",
    device="cuda"
)

print("Dataset size =", len(dataset))
latents, label = dataset[0]
print("latents.shape =", latents.shape, "label =", label)

# 批量读取
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for latents_batch, labels_batch in loader:
    print("Batch latents:", latents_batch.shape)
    print("Batch labels:", labels_batch.shape)
    break