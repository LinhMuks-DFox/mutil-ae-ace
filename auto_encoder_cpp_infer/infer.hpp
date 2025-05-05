// infer.hpp (合并声明和实现)
#ifndef INFER_HPP
#define INFER_HPP

#include <iostream> // 用于日志/调试输出

#include <onnxruntime_cxx_api.h> // ONNX Runtime Header
#include <stdexcept>
#include <string>
#include <vector>

class InferenceEngine {
private:
  // --- 私有成员变量 ---
  Ort::Env env_;
  Ort::SessionOptions session_options_; // 保留 options 成员，方便后续查询或扩展
  Ort::Session session_{nullptr};       // 使用成员初始化列表确保初始化
  Ort::AllocatorWithDefaultOptions allocator_; // 用于管理内部字符串分配
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_from_model_; // 存储从模型读到的输入形状

  // --- 私有辅助函数 (直接在类内实现，效果类似 inline) ---
  void setupSessionOptions(/* 可以添加参数来选择 EP */) {
    session_options_.SetIntraOpNumThreads(1); // 设置线程数 (示例)
    // 设置图优化级别
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // 或 ORT_ENABLE_ALL

    // --- 在这里添加 Execution Provider ---
    // 默认情况下，ORT 会使用 CPU EP。
    // 为了性能，你可能想显式添加或优先选择其他 EP
    // 例如，如果需要 GPU (需要编译 ORT for CUDA 并链接相应库):
    // #ifdef USE_CUDA
    // try {
    //     OrtCUDAProviderOptions cuda_options{};
    //     session_options_.AppendExecutionProvider_CUDA(cuda_options);
    //     std::cout << "InferenceEngine::setupSessionOptions: Appended CUDA
    //     Execution Provider." << std::endl;
    // } catch (const Ort::Exception& e) {
    //     std::cerr << "Warning: Failed to append CUDA EP: " << e.what() <<
    //     std::endl;
    // }
    // #endif
    // 其他 EP 类似添加 (CoreML, NNAPI, TensorRT, OpenVINO etc.)

    std::cout << "InferenceEngine::setupSessionOptions: Configured."
              << std::endl;
  }

  void getInputOutputInfo() {
    size_t num_input_nodes = session_.GetInputCount();
    size_t num_output_nodes = session_.GetOutputCount();
    if (num_input_nodes == 0 || num_output_nodes == 0) {
      throw std::runtime_error(
          "Error: Model has no inputs or outputs according to ONNX Runtime.");
    }
    // 假设我们的模型只有一个输入和一个输出
    if (num_input_nodes != 1 || num_output_nodes != 1) {
      // 打印警告但继续使用第一个，或者可以抛出错误
      std::cerr << "Warning: Model has " << num_input_nodes << " inputs and "
                << num_output_nodes
                << " outputs. Engine currently expects 1 input/output."
                << std::endl;
    }

    // 获取第一个输入的信息
    Ort::AllocatedStringPtr input_name_alloc =
        session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_alloc.get(); // 保存输入节点名称
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    if (!input_tensor_info) {
      throw std::runtime_error(
          "Error: Could not get input tensor type and shape info.");
    }
    input_shape_from_model_ =
        input_tensor_info.GetShape(); // 保存模型定义的输入形状

    // 获取第一个输出的信息
    Ort::AllocatedStringPtr output_name_alloc =
        session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name_alloc.get(); // 保存输出节点名称

    std::cout << "InferenceEngine::getInputOutputInfo: Input='" << input_name_
              << "', Output='" << output_name_ << "'" << std::endl;
    std::cout << "  Model Input Shape: [";
    for (size_t i = 0; i < input_shape_from_model_.size(); ++i)
      std::cout << input_shape_from_model_[i]
                << (i == input_shape_from_model_.size() - 1 ? "" : ", ");
    std::cout << "]" << std::endl;
  }

public:
  // --- 构造函数 ---
  // explicit 关键字防止隐式转换
  explicit InferenceEngine(const std::string &model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxCppInference_Env"), // 初始化环境
        session_options_(), // 初始化会话选项
        session_(nullptr),  // 初始化会话指针
        allocator_()        // 初始化分配器
  {
    std::cout << "InferenceEngine: Initializing for model: " << model_path
              << std::endl;
    if (model_path.empty()) {
      throw std::invalid_argument("Model path cannot be empty.");
    }
    // 检查文件是否存在 (可选但推荐)
    // std::ifstream f(model_path.c_str());
    // if (!f.good()) {
    //     throw std::runtime_error("Model file not found or not readable: " +
    //     model_path);
    // }

    try {
      // 1. 配置会话选项 (包括 EP)
      setupSessionOptions();

      // 2. 创建会话并加载模型
      std::cout << "InferenceEngine: Creating session..." << std::endl;
// 处理 Windows 路径的宽字符需求
#ifdef _WIN32
      std::wstring model_path_w =
          std::wstring(model_path.begin(), model_path.end());
      session_ = Ort::Session(env_, model_path_w.c_str(), session_options_);
#else
      session_ = Ort::Session(env_, model_path.c_str(), session_options_);
#endif
      std::cout << "InferenceEngine: Session created." << std::endl;

      // 3. 获取输入输出信息
      getInputOutputInfo();
      std::cout << "InferenceEngine: Initialization complete." << std::endl;

    } catch (const Ort::Exception &e) {
      // 捕获并重新抛出或记录 ONNX Runtime 特定异常
      std::cerr
          << "InferenceEngine Error during initialization (Ort::Exception): "
          << e.what() << std::endl;
      throw std::runtime_error("Failed to initialize ONNX Runtime session: " +
                               std::string(e.what()));
    } catch (const std::exception &e) {
      // 捕获其他标准异常
      std::cerr << "Standard Error during initialization: " << e.what()
                << std::endl;
      throw; // Re-throw
    }
  }

  // --- 析构函数 ---
  // 由于成员变量使用了 RAII (Resource Acquisition Is Initialization)，
  // Ort::Env, Ort::Session, Ort::SessionOptions 等会自动管理资源，
  // 通常不需要显式析构函数。
  ~InferenceEngine() {
    std::cout << "InferenceEngine: Destroyed." << std::endl;
  }

  // --- 核心推理方法 ---
  std::vector<float> run(const std::vector<float> &input_audio) {
    if (!session_) {
      throw std::runtime_error("InferenceEngine session is not initialized.");
    }
    if (input_audio.empty()) {
      std::cerr << "Warning: Input audio vector is empty." << std::endl;
      // 根据需要可以返回空向量或抛出错误
      return {};
    }

    // 1. 准备输入张量
    int64_t batch_size = 1; // 假设批处理大小为 1
    int64_t sequence_length = static_cast<int64_t>(input_audio.size());
    std::vector<int64_t> input_shape = {batch_size,
                                        sequence_length}; // 本次运行的实际形状

    // 检查输入长度是否合理 (可选)
    // size_t model_input_length = (input_shape_from_model_.size() > 1 &&
    // input_shape_from_model_[1] > 0) ? input_shape_from_model_[1] : 0; if
    // (model_input_length > 0 && sequence_length != model_input_length) {
    //     std::cerr << "Warning: Input audio length (" << sequence_length
    //               << ") does not match model expected length (" <<
    //               model_input_length
    //               << "). Ensure model handles dynamic time axis." <<
    //               std::endl;
    // }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 创建输入 Ort::Value。注意 CreateTensor 需要非 const 数据指针。
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(
            input_audio.data()), // 强制转换，确保 ORT 不会修改输入
        input_audio.size(),      // 元素总数
        input_shape.data(),      // 形状数组指针
        input_shape.size()       // 形状维度数
    );

    // 2. 准备输入/输出名称数组 (从成员变量获取)
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};

    // 3. 执行推理
    std::vector<Ort::Value> output_tensors;
    try {
      output_tensors = session_.Run(Ort::RunOptions{nullptr}, // 默认运行选项
                                    input_names,
                                    &input_tensor, // 指向输入张量数组的指针
                                    1,             // 输入张量数量
                                    output_names,
                                    1 // 期望的输出张量数量
      );
    } catch (const Ort::Exception &e) {
      std::cerr << "InferenceEngine Error during Run: " << e.what()
                << std::endl;
      throw std::runtime_error("ONNX Runtime inference failed: " +
                               std::string(e.what()));
    }

    // 4. 处理输出张量
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
      throw std::runtime_error(
          "Error: Inference returned invalid or empty output tensor.");
    }

    Ort::Value &output_tensor = output_tensors[0];
    // 假设输出是 float 类型，如果不是需要修改 T
    float *output_data = output_tensor.GetTensorMutableData<float>();
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    if (!output_tensor_info) {
      throw std::runtime_error(
          "Error: Could not get output tensor type and shape info.");
    }
    size_t output_data_size = output_tensor_info.GetElementCount();

    // 将结果复制到 std::vector 并返回
    return std::vector<float>(output_data, output_data + output_data_size);
  }

  // --- Getter 方法 ---
  const std::string &getInputName() const { return input_name_; }

  const std::string &getOutputName() const { return output_name_; }

  const std::vector<int64_t> &getInputShape() const {
    // 返回模型定义的形状（可能包含 -1 表示动态轴）
    return input_shape_from_model_;
  }
};

#endif // INFER_HPP