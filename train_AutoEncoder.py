import os
import sys
import random
import yaml
import smtplib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from email.message import EmailMessage
from torch.utils.data import DataLoader, random_split
import tqdm
import torchinfo
import torchaudio.transforms as T
from lib.AudioSet.transform import TimeSequenceLengthFixer, SoundTrackSelector
import lib.MuxkitTools.dataset_tools.CachableDataset as mk_cachedata

import src.AutoEncoder
import src.AudioSetForMakingAutoEncoder
import src.ToLogMelSpectrogram

class ToDevice(nn.Module):
    """将输入Tensor迁移到指定设备的简单Module封装。"""
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)

class AutoEncoderTrainer:
    def __init__(self, 
                 hyperpara_path="auto_encoder_hyperpara.yml",
                 email_config_path="email_config.yml",
                 other_configs_path="other_configs.yml",
                 checkpoint_dir="./auto_encoder_checkpoints"):

        # 读取超参数
        with open(hyperpara_path, "r") as f:
            self.hyper_parameter = yaml.safe_load(f)
        with open(email_config_path, "r") as f:
            self.email_config = yaml.safe_load(f)
        with open(other_configs_path, "r") as f:
            other_configs = yaml.safe_load(f)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 设备
        self.device = self.hyper_parameter["TrainProcessControl"]["device"]

        # 数据预处理流水线
        self.pipeline = nn.Sequential(
            SoundTrackSelector(self.hyper_parameter["SoundTrackSelector"]['mode']),
            ToDevice(self.device),
            T.Resample(**self.hyper_parameter["Resample"]),
            TimeSequenceLengthFixer(**self.hyper_parameter["TimeSequenceLengthFixer"]),
            ToLogMelSpectrogram.ToLogMelSpectrogram(**self.hyper_parameter["ToLogMelSpectrogram.py"])
        ).to(self.device)

        # 数据集加载
        trainset, evalset = Dataset.AudiosetForMakingAutoEncoder.from_yaml(other_configs)
        trainset.transform = self.data_preprocess
        evalset.transform = self.data_preprocess

        # 构建子集、拆分、缓存
        subset_size = 1000
        val_test_split = 500
        subset = torch.utils.data.Subset(evalset, torch.randperm(len(evalset))[:subset_size].tolist())
        self.validate_set, self.test_set = random_split(subset, [val_test_split, subset_size - val_test_split])

        self.trainset = mk_cachedata.CacheableDataset(trainset, **self.hyper_parameter["CacheableDataset"])
        self.validate_set = mk_cachedata.CacheableDataset(self.validate_set, **self.hyper_parameter["CacheableDataset"])
        self.test_set = mk_cachedata.CacheableDataset(self.test_set, **self.hyper_parameter["CacheableDataset"])

        # 初始化AutoEncoder模型
        self.model = AutoEncoder.AutoEncoder(**self.hyper_parameter["AutoEncoder"]).to(self.device)
        with torch.no_grad():
            data0, _ = self.trainset[0]
            print("Feeded Data shape: ", data0.shape)
            torchinfo.summary(self.model, input_data=data0.unsqueeze(0))
        # 优化器和调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.hyper_parameter["TrainProcessControl"]["LearningRate"]),
            weight_decay=float(self.hyper_parameter["TrainProcessControl"]["Optimizer"]["WeightDecay"])
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.hyper_parameter["TrainProcessControl"]["Scheduler"]["patience"],
        )

        # 其他训练参数
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.validate_losses = []
        self.lr_list = []

        # 训练超参
        self.batch_size = self.hyper_parameter["TrainProcessControl"]["BatchSize"]
        self.epochs = self.hyper_parameter["TrainProcessControl"]["Epoch"]
        self.val_interval = self.hyper_parameter["TrainProcessControl"].get("val_interval", 1)
        self.save_interval = self.hyper_parameter["TrainProcessControl"].get("save_interval", 5)

        # 数据加载器
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.validate_set, batch_size=self.batch_size)

    @torch.no_grad()
    def data_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """用预定义的 pipeline 对音频做预处理。"""
        return self.pipeline(x)

    def one_step_loss(self, x, x_hat):
        """计算单次前向传播的loss。兼容多维数据"""
        if x.dim() > 3:
            batch_size, _, h, w = x.shape
            x = x.view(batch_size, h, w)
        return self.loss_function(x, x_hat)

    def one_epoch_train(self):
        """单个Epoch的训练流程。"""
        epoch_losses = []
        for batch_data, _ in tqdm.tqdm(self.train_loader):
            batch_data = batch_data.to(self.device)

            self.optimizer.zero_grad()
            x_hat = self.model(batch_data)
            loss = self.one_step_loss(batch_data, x_hat)
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())
        return float(np.mean(epoch_losses))

    @torch.no_grad()
    def validate(self):
        """验证流程，返回本次验证Loss均值。"""
        self.model.eval()
        val_epoch_losses = []
        for batch_data, _ in tqdm.tqdm(self.val_loader):
            batch_data = batch_data.to(self.device)
            x_hat = self.model(batch_data)
            loss = self.one_step_loss(batch_data, x_hat)
            val_epoch_losses.append(loss.item())
        mean_val_loss = float(np.mean(val_epoch_losses))
        self.model.train()
        return mean_val_loss

    def save_checkpoint(self, epoch=None):
        """保存模型checkpoint，同时保存训练过程中的loss、验证loss和学习率数据到独立子文件夹中。"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 创建一个基于当前时间的子文件夹
        subfolder = os.path.join(self.checkpoint_dir, f"checkpoint_{now}")
        os.makedirs(subfolder, exist_ok=True)
        
        filename = f"epoch_{epoch}" if epoch is not None else "final"
        filename += ".pt"
        save_path = os.path.join(subfolder, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'validate_losses': self.validate_losses,
            'lr_list': self.lr_list,
        }, save_path)
        
        print(f"[✓] Checkpoint saved: {save_path}")

    def plot_losses(self, save_path="loss_curve.png"):
        """绘制训练和验证Loss曲线。"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        # 验证曲线：间隔self.val_interval记录一次
        val_x = np.arange(self.val_interval - 1, self.epochs, self.val_interval)
        plt.plot(val_x, self.validate_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Loss curve saved to {save_path}")

    def plot_learning_rate(self, save_path="lr_curve.png"):
        """绘制学习率变化曲线。"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.lr_list, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Learning rate curve saved to {save_path}")

    @torch.no_grad()
    def show_reconstruction(self, dataset, save_path="reconstruction.png"):
        """随机选取一个样本进行前向重建并可视化对比。"""
        self.model.eval()
        sample_idx = random.randint(0, len(dataset) - 1)
        data, _ = dataset[sample_idx]  # 注意dataset返回的是 (data, label)
        data = data.unsqueeze(0).to(self.device)

        reconstructed = self.model(data)

        original_np = data.squeeze().cpu().numpy()
        reconstructed_np = reconstructed.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(original_np, aspect="auto", origin="lower")
        axs[0].set_title("Original Log-Mel")
        axs[1].imshow(reconstructed_np, aspect="auto", origin="lower")
        axs[1].set_title("Reconstructed Log-Mel")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Reconstruction plot saved to {save_path}")

    def send_mail(self, subject, body):
        """发送简单的通知邮件。"""
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.email_config["email"]["from"]
            msg["To"] = self.email_config["email"]["to"]
            msg.set_content(body)

            with smtplib.SMTP(self.email_config["email"]["smtp_host"], self.email_config["email"]["smtp_port"]) as smtp:
                smtp.starttls()
                smtp.login(self.email_config["email"]["from"], self.email_config["email"]["password"])
                smtp.send_message(msg)
            print("[✓] Notification email sent.")
        except Exception as e:
            print(f"[!] Failed to send notification email: {e}")

    def run(self):
        """执行整个训练流程。包括训练、验证、可视化和邮件通知等。"""
        try:
            for epoch in range(self.epochs):
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_list.append(current_lr)

                print(f"\nEpoch {epoch + 1}/{self.epochs}, LR={current_lr}")

                # 训练
                train_loss = self.one_epoch_train()
                self.train_losses.append(train_loss)
                print(f"Train Loss: {train_loss:.6f}")

                # 验证
                if (epoch + 1) % self.val_interval == 0:
                    val_loss = self.validate()
                    self.validate_losses.append(val_loss)
                    print(f"Validation Loss: {val_loss:.6f}")
                    self.scheduler.step(val_loss)

                # 保存checkpoint
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch + 1)

            # 训练结束后，画出loss曲线和学习率曲线
            self.plot_losses()
            self.plot_learning_rate()

            # 随机可视化一次重建
            self.show_reconstruction(self.validate_set)

            # 发送邮件通知
            if len(self.train_losses) > 0:
                final_train_loss = self.train_losses[-1]
                final_val_loss = self.validate_losses[-1] if len(self.validate_losses) > 0 else None
                body_msg = (
                    f"Training completed successfully.\n"
                    f"Final Training Loss: {final_train_loss:.6f}\n"
                )
                if final_val_loss is not None:
                    body_msg += f"Final Validation Loss: {final_val_loss:.6f}\n"
                self.send_mail("AutoEncoder Training Finished", body_msg)

        except KeyboardInterrupt:
            print("\n[!] 收到 KeyboardInterrupt，正在保存checkpoint和曲线后退出...")
            self.save_checkpoint(epoch="KeyboardInterrupt")
            self.plot_losses("loss_curve_keyboard_interrupt.png")
            self.plot_learning_rate("lr_curve_keyboard_interrupt.png")
            sys.exit(1)

        except Exception as e:
            print("[!] 发生异常：", e)
            print("[!] 正在保存checkpoint和曲线，以便后续排查...")
            self.save_checkpoint(epoch="Exception")
            self.plot_losses("loss_curve_exception.png")
            self.plot_learning_rate("lr_curve_exception.png")
            raise e

if __name__ == "__main__":
    trainer = AutoEncoderTrainer(
        hyperpara_path="configs/auto_encoder_hyperpara.yml",
        email_config_path="configs/email_config.yml",
        other_configs_path="configs/dataset_info.yml"
    )
    trainer.run()