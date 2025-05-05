import torch
import torch.nn as nn
import torchaudio # 用于获取 Mel 滤波器组，可替换为预计算加载
import yaml
import torchinfo
from collections import OrderedDict
import traceback
import os

# --- 假设 AutoEncoder 类在 src/AutoEncoder.py 中定义 ---
# 请确保此导入路径相对于你的工作目录是正确的
try:
    from src.AutoEncoder import AutoEncoder
except ImportError:
    print("Error: Failed to import AutoEncoder from src.AutoEncoder")
    print("Please ensure src/AutoEncoder.py exists and is in the Python path.")
    exit()
# ------------------------------------------------------

class ManualLogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0, top_db=None, mel_norm='slaney', mel_scale="htk", f_min=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.top_db = top_db

        try:
            mel_filters = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=sample_rate / 2.0,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm=mel_norm if mel_norm != "None" else None,
                mel_scale=mel_scale,
            )
            self.register_buffer('mel_filters', mel_filters)
            print(f"Mel filters shape: {self.mel_filters.shape}")
        except Exception as e:
            print(f"Error creating Mel filterbanks using torchaudio: {e}")
            raise

        self.register_buffer('window', torch.hann_window(self.n_fft))

    def forward(self, waveform: torch.Tensor):
        # 1. STFT - *** 修改这里: return_complex=False ***
        # 这将返回一个形状为 (..., F, T, 2) 的实数张量, 最后一维是 [实部, 虚部]
        stft_real_imag = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=False # <--- 主要改动
        ) # Shape: (..., F, T, 2)

        # 2. Calculate magnitude from real/imag parts
        # 提取实部和虚部
        real_part = stft_real_imag[..., 0] # Shape: (..., F, T)
        imag_part = stft_real_imag[..., 1] # Shape: (..., F, T)
        # 计算模长
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2)) # Shape: (..., F, T)

        # --- 后续步骤不变 ---
        # Calculate power spectrum
        power_spec = magnitude.pow(self.power) # Shape: (..., F, T)

        # Apply Mel filterbanks
        mel_spec = torch.matmul(power_spec.transpose(-1, -2), self.mel_filters).transpose(-1, -2)

        # Convert to dB
        amin = torch.tensor(1e-10, device=mel_spec.device, dtype=mel_spec.dtype)
        log_spec = 10.0 * torch.log10(torch.maximum(mel_spec, amin))

        if self.top_db is not None:
            max_val = torch.amax(log_spec, dim=(-2, -1), keepdim=True)
            log_spec = torch.maximum(log_spec, max_val - self.top_db)

        return log_spec
# --- 部署模块，使用手动实现的预处理 ---
class AutoEncoderDeploymentInferenceModule(torch.nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        super().__init__()

        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        with open(config_path, "r") as f:
            hyper = yaml.safe_load(f)

            # --- 加载 AutoEncoder ---
            ae_hyper = hyper.get("AutoEncoder")
            if ae_hyper is None:
                 raise ValueError("Config missing 'AutoEncoder' section")
            print("Loading AutoEncoder model...")
            self.auto_encoder = AutoEncoder.from_structure_hyper_and_checkpoint(
                ae_hyper, checkpoint_path, device
            )
            # 确保 AutoEncoder 的懒加载层已初始化 (非常重要!)
            # 如果 from_structure_hyper_and_checkpoint 没有处理，需要在这里显式初始化
            # 例如:
            # if "AutoEncoderInitDummyInput" not in ae_hyper:
            #      print("Warning: Manually triggering AutoEncoder lazy init (assuming it's needed)")
            #      try:
            #         # Need a plausible input shape for the *encoder* part
            #         # Example: (Batch, n_mels, Time_after_preprocessing)
            #         dummy_mel_shape = (1, ae_hyper.get('n_mel', 80), 256) # Adjust T dim as needed
            #         dummy_mel_input = torch.randn(*dummy_mel_shape, device=device)
            #         with torch.no_grad():
            #             self.auto_encoder.encode(dummy_mel_input)
            #         print("AutoEncoder lazy init triggered.")
            #      except Exception as init_e:
            #          print(f"ERROR during manual lazy init: {init_e}")
            #          print("Proceeding anyway, but ONNX export might fail if lazy layers aren't ready.")


            # --- 使用手动实现的 Mel 谱图模块 ---
            mel_params = hyper.get("ToLogMelSpectrogram")
            if mel_params is None:
                raise ValueError("Config missing 'ToLogMelSpectrogram' section")

            # 提取 AmplitudeToDB 的参数 (如果存在于配置中)
            # 假设 top_db 参数在 ToLogMelSpectrogram 或一个独立的 AmplitudeToDB 部分
            amp_to_db_params = hyper.get("AmplitudeToDB", {}) # Or adjust if nested differently
            top_db_value = amp_to_db_params.get("top_db") # Might be None

            print("Initializing ManualLogMelSpectrogram...")
            self.to_mel = ManualLogMelSpectrogram(
                sample_rate=mel_params.get("sample_rate", 16000),
                n_fft=mel_params.get("n_fft", 400),
                hop_length=mel_params.get("hop_length", 160),
                n_mels=mel_params.get("n_mels", 80),
                power=mel_params.get("power", 2.0), # Assuming power might be in config
                top_db=top_db_value, # Pass top_db if found
                mel_norm=mel_params.get("mel_norm", 'slaney'), # Add other relevant params
                mel_scale=mel_params.get("mel_scale", 'htk'),
                f_min=mel_params.get("f_min", 0.0)
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # 端到端推理：音频 -> LogMel -> 编码 -> 潜在向量
        ret = self.to_mel(x)
        ret = self.auto_encoder.encode(ret)
        return ret

    def get_seq(self) -> torch.nn.Sequential:
        # 返回包含预处理和编码器的序列模型
        # 确保调用此方法前，auto_encoder 的懒加载层已初始化
        print("Creating Sequential model for export...")
        try:
             encoder_part = self.auto_encoder.get_encoder()
             print("Successfully got encoder part from AutoEncoder.")
             return torch.nn.Sequential(OrderedDict([
                 ("Preprocessing", self.to_mel),
                 ("AutoEncoder-Encoder", encoder_part),
             ]))
        except Exception as e:
             print(f"Error getting encoder part: {e}")
             print("This might happen if lazy layers in AutoEncoder were not initialized.")
             raise

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 配置区 ---
    CONFIG_FILE = "configs/auto_encoder_hyperpara.yml"
    CHECKPOINT_FILE = "autoencoder_checkpoint/run_20250413_155634/checkpoints/epoch_300.pt"
    ONNX_OUTPUT_FILE = "preproc_encoder.onnx"
    # 尝试 Opset 17，因为 STFT 需要它，希望手动实现解决了复数问题
    # 如果仍然失败，可以尝试降低 Opset，但这可能不支持 STFT 导出
    TARGET_OPSET = 17
    # 输入音频长度（例如 5 秒）
    AUDIO_LENGTH_SECONDS = 5
    SAMPLE_RATE = 16000 # 从配置中读取或硬编码，需要与模型训练时一致
    # -----------

    print(f"Using Config: {CONFIG_FILE}")
    print(f"Using Checkpoint: {CHECKPOINT_FILE}")
    print(f"Output ONNX file: {ONNX_OUTPUT_FILE}")
    print(f"Target Opset: {TARGET_OPSET}")

    # 检查文件是否存在
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Config file not found at {CONFIG_FILE}")
        exit()
    if not os.path.exists(CHECKPOINT_FILE):
         print(f"ERROR: Checkpoint file not found at {CHECKPOINT_FILE}")
         exit()

    try:
        print("\nInitializing deployment module...")
        # 使用 "cpu" 进行模型加载和导出通常更安全
        deployment_module = AutoEncoderDeploymentInferenceModule(CONFIG_FILE, CHECKPOINT_FILE, device="cpu")
        # 获取用于导出的序列模型，并设置为评估模式
        ae_export_model = deployment_module.get_seq().cpu().eval()
        print("Deployment module initialized and sequential model obtained.")

        # 定义虚拟输入
        sequence_length = SAMPLE_RATE * AUDIO_LENGTH_SECONDS
        dummy_input = torch.randn(1, sequence_length) # (Batch=1, Time)
        input_shape = dummy_input.shape
        print(f"\nUsing dummy input shape for export: {input_shape}")

        # 可选：运行 torchinfo 摘要
        try:
            print("Running torchinfo summary...")
            # 注意：summary 可能会执行模型，确保懒加载层已初始化
            torchinfo.summary(ae_export_model, input_shape)
            print("Summary finished.")
        except Exception as e:
            print(f"torchinfo summary failed (might be ok, continuing export): {e}\n")

        # 定义导出参数
        input_names = ['audio_input']
        output_names = ['latent_output']
        dynamic_axes = {
            'audio_input': {0: 'batch_size', 1: 'time'},
            'latent_output': {0: 'batch_size'}
        }

        print(f"\nAttempting to export model to ONNX (Opset {TARGET_OPSET})...")
        torch.onnx.export(
            ae_export_model,
            dummy_input,
            ONNX_OUTPUT_FILE,
            export_params=True,
            opset_version=TARGET_OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print("\n-----------------------------------------")
        print(f" ONNX export completed successfully! ")
        print(f" Model saved to: {ONNX_OUTPUT_FILE}")
        print("-----------------------------------------")
        print("\nReminder: Validate the exported ONNX model using onnx.checker and onnxruntime.")

    except FileNotFoundError as e:
         print(f"\nERROR: File not found during initialization. {e}")
    except ImportError as e:
         print(f"\nERROR: Could not import necessary modules. {e}")
    except KeyError as e:
         print(f"\nERROR: Missing key in config file '{CONFIG_FILE}'. Key: {e}")
    except ValueError as e:
         print(f"\nERROR: Configuration value error. {e}")
    except RuntimeError as e:
        print(f"\nERROR during model processing or export: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
    except Exception as e:
        print(f"\nAn unexpected ERROR occurred: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")