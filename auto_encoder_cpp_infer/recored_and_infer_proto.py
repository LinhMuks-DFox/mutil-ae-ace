import sounddevice as sd
import numpy as np
import onnxruntime as ort
import time
import sys
import os
import traceback
import soundfile as sf

# --- 配置参数 ---
SAMPLE_RATE = 16000
DURATION = 5
CHANNELS = 1
ONNX_MODEL_PATH = "preproc_encoder.onnx"
OUTPUT_DTYPE = np.float32
OUTPUT_WAV_PATH = "recorded_audio.wav"
OUTPUT_LATENT_PATH = "latent_vector.npy" # <--- 定义潜在向量输出文件名 (.npy)
# ---

# ... (函数 record_audio 和 run_onnx_inference 保持不变) ...

def record_audio(duration: int, samplerate: int, channels: int, dtype: np.dtype = OUTPUT_DTYPE, save_path: str = OUTPUT_WAV_PATH) -> np.ndarray:
    # ... (此函数内部不变，它仍然返回 onnx_input) ...
    num_samples = int(duration * samplerate)
    print(f"\n准备录音...")
    print(f"将在 2 秒后开始录制 {duration} 秒音频 ({samplerate} Hz)...")
    time.sleep(2)
    print("开始录音!")

    try:
        recording_raw = sd.rec(num_samples, samplerate=samplerate, channels=channels, dtype='float32', blocking=True)
        print("录音结束.")

        # --- 保存录音文件 ---
        try:
            print(f"正在将录音保存到 {save_path} ...")
            sf.write(save_path, recording_raw, samplerate)
            print(f"录音已成功保存到: {save_path}")
        except Exception as e:
            print(f"\n错误: 保存录音文件失败: {e}")
        # -----------------------

        # --- 准备 ONNX 输入数据 ---
        if recording_raw.dtype != dtype:
             recording_for_onnx = recording_raw.astype(dtype)
        else:
             recording_for_onnx = recording_raw

        print(f"原始录音数组形状 (用于 ONNX): {recording_for_onnx.shape}")
        if channels == 1:
            recording_flat = recording_for_onnx.flatten()
        else:
            print(f"警告: 录制到 {channels} 个声道, 将只使用第一个声道进行推理。")
            recording_flat = recording_for_onnx[:, 0].flatten()

        onnx_input = np.expand_dims(recording_flat, axis=0)
        print(f"处理后的 ONNX 输入形状: {onnx_input.shape}, 类型: {onnx_input.dtype}")
        return onnx_input

    except sd.PortAudioError as e:
        print(f"PortAudio 错误 (请检查音频设备): {e}")
        raise
    except Exception as e:
        print(f"录音过程中发生错误: {e}")
        raise

def run_onnx_inference(model_path: str, input_data: np.ndarray) -> np.ndarray:
    # ... (此函数内部不变，它仍然返回 latent_vector) ...
    print(f"\n正在加载 ONNX 模型: {model_path}")
    try:
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX Runtime Session 已创建，使用 Provider: {session.get_providers()}")

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"模型输入节点名: {input_name}")
        print(f"模型输出节点名: {output_name}")

        input_feed = {input_name: input_data}

        print(f"开始推理，输入数据形状: {input_data.shape}...")
        start_time = time.time()
        results = session.run([output_name], input_feed)
        end_time = time.time()
        print(f"推理完成，耗时: {end_time - start_time:.3f} 秒.")

        latent_vector = results[0]
        return latent_vector

    except ort.OrtLoadError as e:
        print(f"错误: 加载 ONNX 模型失败: {e}")
        raise
    except Exception as e:
        print(f"ONNX 推理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误: 找不到 ONNX 模型文件 '{ONNX_MODEL_PATH}'")
        sys.exit(1)

    try:
        # 1. 录制音频 (并保存到 WAV)
        audio_input_for_onnx = record_audio(DURATION, SAMPLE_RATE, CHANNELS, save_path=OUTPUT_WAV_PATH)

        # 2. 执行 ONNX 推理
        latent_output_vector = run_onnx_inference(ONNX_MODEL_PATH, audio_input_for_onnx)

        # 3. 显示结果
        print("\n--- 推理结果 ---")
        print(f"输出潜在向量形状: {latent_output_vector.shape}")
        print(f"输出潜在向量类型: {latent_output_vector.dtype}")
        print("潜在向量 (前 10 个元素):")
        print(latent_output_vector[0, :10])
        print("-----------------")

        # --- 4. 将推理结果 (潜在向量) 保存到文件 ---
        try:
            print(f"\n正在将潜在向量保存到文件: {OUTPUT_LATENT_PATH} ...")
            # 使用 NumPy 的 save 函数，保存为 .npy 格式
            np.save(OUTPUT_LATENT_PATH, latent_output_vector) # <--- 保存 NumPy 数组
            print(f"潜在向量已成功保存到: {OUTPUT_LATENT_PATH}")

            # --- (可选) 如果你确实需要 .pt 格式 ---
            # # 你需要先 import torch
            # try:
            #     import torch
            #     output_pt_path = "latent_vector.pt"
            #     print(f"正在将潜在向量转换为 PyTorch Tensor 并保存到: {output_pt_path} ...")
            #     latent_tensor = torch.from_numpy(latent_output_vector)
            #     torch.save(latent_tensor, output_pt_path)
            #     print(f"PyTorch Tensor 已成功保存到: {output_pt_path}")
            # except ImportError:
            #     print("错误: 需要安装 PyTorch ('pip install torch') 才能保存为 .pt 文件。")
            # except Exception as pt_e:
            #     print(f"错误: 保存为 .pt 文件失败: {pt_e}")
            # ---------------------------------------

        except Exception as e:
            print(f"\n错误: 保存潜在向量失败: {e}")
        # ------------------------------------------

        print(f"\n录音文件已保存为 '{OUTPUT_WAV_PATH}' (如果保存成功)。")
        print("脚本执行完毕。")

    except ImportError as e:
        print("\n错误: 找不到必要的库。")
        print("请运行: pip install sounddevice onnxruntime numpy soundfile")
        print("在 Linux/Raspberry Pi 上可能还需要: sudo apt-get update && sudo apt-get install libsndfile1")
        # 如果取消注释了 torch.save: print("如果需要保存为 .pt, 还需要: pip install torch")
    except Exception as e:
        print(f"\n脚本执行过程中发生错误: {e}")
        # traceback.print_exc() # 取消注释以查看完整错误堆栈
        sys.exit(1)