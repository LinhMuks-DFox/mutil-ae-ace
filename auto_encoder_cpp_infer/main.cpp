#include "infer.hpp" // 包含推理引擎头文件
#include <algorithm> // 用于 std::generate
#include <chrono>    // 用于计时 (可选)

#include <iostream>
#include <random> // 用于生成随机数

#include <string> // 用于 std::string
#include <vector>

int main(int argc, char *argv[]) {
  // 现在只需要模型路径作为参数
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path.onnx>" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::cout << "Model Path: " << model_path << std::endl;

  try {
    // --- 1. 生成随机音频数据 ---
    const int sample_rate = 16000;
    const int duration_seconds = 5;
    const size_t num_samples =
        static_cast<size_t>(sample_rate) * duration_seconds;

    std::cout << "Generating random audio data..." << std::endl;
    std::cout << "  Sample Rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "  Number of Samples: " << num_samples << std::endl;

    std::vector<float> audio_data(num_samples);

    // 使用 <random> 生成高质量随机数
    std::random_device rd;  // 用于获取随机种子 (如果可用)
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
    // 生成 [-1.0, 1.0] 范围内的均匀分布浮点数
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);

    // 使用 std::generate 填充 vector
    std::generate(audio_data.begin(), audio_data.end(),
                  [&]() { return distrib(gen); });
    // 或者使用传统循环:
    // for (float& sample : audio_data) {
    //     sample = distrib(gen);
    // }

    std::cout << "Random audio data generated." << std::endl;
    // 打印一小部分生成的样本值 (可选)
    // std::cout << "  First 5 samples: ";
    // for(size_t i = 0; i < std::min((size_t)5, audio_data.size()); ++i) {
    //     std::cout << audio_data[i] << " ";
    // }
    // std::cout << std::endl;
    // ------------------------------------

    // 2. 创建推理引擎实例
    std::cout << "Initializing inference engine..." << std::endl;
    InferenceEngine engine(
        model_path); // 假设 InferenceEngine 在 infer.hpp 中定义
    std::cout << "Inference engine initialized." << std::endl;
    std::cout << "  Model Input Name: " << engine.getInputName() << std::endl;
    std::cout << "  Model Output Name: " << engine.getOutputName() << std::endl;

    // 3. 执行推理
    std::cout << "Running inference with random data..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> latent_vector = engine.run(audio_data); // 将随机数据传入
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