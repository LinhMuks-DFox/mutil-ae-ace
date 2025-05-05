#include "infer.hpp" // 包含推理引擎头文件
#include <chrono>    // 用于计时 (可选)
#include <iostream>
#include <vector>

// --- 音频加载函数 (需要实现或使用库) ---
// 返回 true 如果加载成功, 音频数据在 out_audio 中
bool loadAudioFile(const std::string &file_path, std::vector<float> &out_audio,
                   int &sample_rate) {
  // 在这里使用音频库 (如 libsndfile, dr_wav 等) 加载 WAV 文件
  // 确保加载为 float 类型，并获取采样率
  // 示例占位符：
  std::cerr << "Warning: Audio loading not implemented. Using dummy data."
            << std::endl;
  sample_rate = 16000;               // 假设采样率
  out_audio.resize(sample_rate * 3); // 3 秒的虚拟数据
  for (size_t i = 0; i < out_audio.size(); ++i)
    out_audio[i] = static_cast<float>(i % 100) / 100.0f - 0.5f;
  if (file_path.empty())
    return false; // 示例性失败
  return true;
}
// ---------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_path.onnx> <audio_file.wav>"
              << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::string audio_path = argv[2];

  std::cout << "Model Path: " << model_path << std::endl;
  std::cout << "Audio Path: " << audio_path << std::endl;

  try {
    // 1. 加载音频
    std::vector<float> audio_data;
    int sample_rate;
    std::cout << "Loading audio file..." << std::endl;
    if (!loadAudioFile(audio_path, audio_data, sample_rate)) {
      std::cerr << "Error: Failed to load audio file: " << audio_path
                << std::endl;
      return 1;
    }
    std::cout << "Audio loaded (" << audio_data.size() / (float)sample_rate
              << " seconds, " << sample_rate << " Hz)." << std::endl;

    // 2. 创建推理引擎实例
    std::cout << "Initializing inference engine..." << std::endl;
    InferenceEngine engine(model_path);
    std::cout << "Inference engine initialized." << std::endl;
    std::cout << "  Model Input Name: " << engine.getInputName() << std::endl;
    std::cout << "  Model Output Name: " << engine.getOutputName() << std::endl;

    // 3. 执行推理
    std::cout << "Running inference..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> latent_vector = engine.run(audio_data);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Inference finished in " << duration.count() << " ms."
              << std::endl;

    // 4. 处理结果
    std::cout << "Inference successful. Latent vector size: "
              << latent_vector.size() << std::endl;
    // 打印部分结果
    std::cout << "Latent vector (first 10 elements): ";
    for (size_t i = 0; i < std::min((size_t)10, latent_vector.size()); ++i) {
      std::cout << latent_vector[i] << " ";
    }
    std::cout << std::endl;

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Standard Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}