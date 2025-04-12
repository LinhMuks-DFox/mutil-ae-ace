import AutoEncoder
import torch.nn as nn
import torch.nn.init as init
import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
import tqdm
import torch
import torchaudio.transforms as T
import Dataset
from lib.AudioSet.transform import TimeSequenceLengthFixer, SoundTrackSelector
import yaml
import numpy as np
import DataTransform
from torch.utils.data import random_split, DataLoader
import lib.MuxkitDeepLearningTools.dataset_tools.CachableDataset as mk_cachedata
import smtplib
from email.message import EmailMessage

class ToDevice(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = "cuda"

    def forward(self, x):
        return x.to(self.device)

with open("hyperpara.yml", "r") as f:
    hyper_parameter = yaml.safe_load(f)
with open("email_config.yml", "r") as f:
    email_config = yaml.safe_load(f)

Device = hyper_parameter["TrainProcessControl"]["device"]

pipeline = nn.Sequential(
    SoundTrackSelector(hyper_parameter["SoundTrackSelector"]['mode']),
    ToDevice(Device),
    T.Resample(**hyper_parameter["Resample"]),
    TimeSequenceLengthFixer(**hyper_parameter["TimeSequenceLengthFixer"]),
    DataTransform.ToLogMelSpectrogram(**hyper_parameter["ToLogMelSpectrogram"]),
).to(Device)

@torch.no_grad()
def data_preprocess(x: torch.Tensor)->torch.Tensor:
    return pipeline(x)
with open("other_configs.yml", "r") as f:
    trainset, evalset = Dataset.AudioSet.from_yaml(yaml.safe_load(f))
trainset.transform=data_preprocess
evalset.transform=data_preprocess

model = AutoEncoder.AutoEncoder(**hyper_parameter["AutoEncoder"]).to(Device)
# from torchinfo import summary
# summary(model, input_size=trainset[0].shape)

optimizer = torch.optim.Adam(model.parameters(),
                       lr=float(hyper_parameter["TrainProcessControl"]["LearningRate"]),
                       weight_decay=float(hyper_parameter["TrainProcessControl"]["Optimizer"]["WeightDecay"]))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=hyper_parameter["TrainProcessControl"]["Scheduler"]["patience"],
)

loss_function = nn.MSELoss()
train_losses = torch.empty(0).to(Device)
validate_losses = torch.empty(0).to(Device)

subset_size = 1000
val_test_split = 500

subset = torch.utils.data.Subset(evalset, torch.randperm(len(evalset))[:subset_size].tolist())
validate_set, test_set = random_split(subset, [val_test_split, subset_size - val_test_split])
trainset = mk_cachedata.CacheableDataset(trainset, **hyper_parameter["CacheableDataset"])
validate_set = mk_cachedata.CacheableDataset(validate_set, **hyper_parameter["CacheableDataset"])
test_set = mk_cachedata.CacheableDataset(test_set, **hyper_parameter["CacheableDataset"])
def one_step_loss(x, x_hat):
    return loss_function(x, x_hat)
def one_epoch_train(dataloader, loss_save_to: torch.Tensor):
    epoch_losses = torch.empty(0, device=Device)
    for batch, _ in tqdm.tqdm(dataloader):
        batch = batch.to(Device)
        optimizer.zero_grad()

        x_hat = model(batch)
        loss = one_step_loss(batch, x_hat)
        loss.backward()
        optimizer.step()

        epoch_losses = torch.cat([epoch_losses, loss.detach().unsqueeze(0)])

    mean_loss = epoch_losses.mean()
    loss_save_to = torch.cat([loss_save_to, mean_loss.unsqueeze(0)])
    return loss_save_to
def save_checkpoint(model, optimizer, epoch=None, save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"checkpoint_{now}.pt"
    save_path = os.path.join(save_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

    print(f"[✓] Checkpoint saved: {save_path}")

batch_size = hyper_parameter["TrainProcessControl"]["BatchSize"]
epochs = hyper_parameter["TrainProcessControl"]["Epoch"]
val_interval = hyper_parameter["TrainProcessControl"].get("val_interval", 1)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validate_set, batch_size=batch_size)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_losses = one_epoch_train(train_loader, train_losses)
    print(f"Train Loss: {train_losses[-1].item():.6f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_losses = torch.empty(0, device=Device)
        with torch.no_grad():
            for batch, _ in tqdm.tqdm(val_loader):
                batch = batch.to(Device)
                x_hat = model(batch)
                loss = one_step_loss(batch, x_hat)
                val_epoch_losses = torch.cat([val_epoch_losses, loss.unsqueeze(0)])
        mean_val_loss = val_epoch_losses.mean()
        validate_losses = torch.cat([validate_losses, mean_val_loss.unsqueeze(0)])
        print(f"Validation Loss: {mean_val_loss.item():.6f}")
        scheduler.step(mean_val_loss)

    model.train()
    if (epoch + 1) % hyper_parameter["TrainProcessControl"].get("save_interval", 5) == 0:
        save_checkpoint(model, optimizer, epoch)

# Plot and save loss curves
def plot_losses(train_losses, validate_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses.cpu().numpy(), label="Train Loss")
    plt.plot(np.arange(val_interval - 1, epochs, val_interval), validate_losses.cpu().numpy(), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"[✓] Loss curve saved to {save_path}")

plot_losses(train_losses, validate_losses)

# Visualize one reconstruction
def show_reconstruction(autoencoder, dataset, save_path="reconstruction.png"):
    autoencoder.eval()
    sample_idx = random.randint(0, len(dataset) - 1)
    original = dataset[sample_idx].unsqueeze(0).to(Device)

    with torch.no_grad():
        reconstructed = autoencoder(original)

    original_np = original.squeeze().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(original_np, aspect="auto", origin="lower")
    axs[0].set_title("Original Log-Mel")
    axs[1].imshow(reconstructed_np, aspect="auto", origin="lower")
    axs[1].set_title("Reconstructed Log-Mel")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Reconstruction plot saved to {save_path}")

show_reconstruction(model, validate_set)

def send_mail(subject, body, config):
    msg = EmailMessage()
    with open("email_config.yml", "r") as f:
        config = yaml.safe_load(f)
    msg["Subject"] = subject
    msg["From"] = config["email"]["from"]
    msg["To"] = config["email"]["to"]
    msg.set_content(body)

    with smtplib.SMTP(config["email"]["smtp_host"], config["email"]["smtp_port"]) as smtp:
        smtp.starttls()
        smtp.login(config["email"]["from"], config["email"]["password"])
        smtp.send_message(msg)

try:
    send_mail(
        subject="AutoEncoder Training Finished",
        body=f"Training completed successfully.\nFinal Training Loss: {train_losses[-1].item():.6f}\nFinal Validation Loss: {validate_losses[-1].item():.6f}",
        config=email_config
    )
    print("[✓] Notification email sent.")
except Exception as e:
    print(f"[!] Failed to send notification email: {e}")
