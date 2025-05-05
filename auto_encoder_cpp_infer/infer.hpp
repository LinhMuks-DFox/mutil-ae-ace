#ifndef INFER_HPP
#define INFER_HPP

#include <memory>                // For smart pointers like unique_ptr
#include <onnxruntime_cxx_api.h> // ONNX Runtime Header
#include <string>
#include <vector>

class InferenceEngine {
public:
  // 构造函数：加载模型，初始化 Session
  // 可以添加参数来控制 EP 的选择，例如 use_gpu=false
  explicit InferenceEngine(const std::string &model_path);

  // 禁用拷贝构造和赋值
  InferenceEngine(const InferenceEngine &) = delete;
  InferenceEngine &operator=(const InferenceEngine &) = delete;

  // 推理函数：接收音频数据，返回潜在向量
  std::vector<float> run(const std::vector<float> &input_audio);

  // (可选) 获取模型输入/输出信息的方法
  const std::vector<int64_t> &getInputShape() const;
  const std::string &getInputName() const;
  const std::string &getOutputName() const;

private:
  // ONNX Runtime 核心对象
  Ort::Env env_;
  Ort::Session session_{nullptr}; // 使用初始化列表初始化为 nullptr

  // 输入/输出信息 (在构造函数中获取并存储)
  Ort::AllocatorWithDefaultOptions allocator_; // 用于管理字符串分配
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_from_model_; // 模型定义的形状
  // 可以添加 output_shape_ 等

  // 辅助函数 (可选, 设为 private)
  void setupSessionOptions(/* params for EPs */); // 配置 SessionOptions
  void getInputOutputInfo();                      // 获取并存储节点信息

  Ort::SessionOptions session_options_; // 将其设为成员，以便在构造时配置
};

#endif // INFER_HPP