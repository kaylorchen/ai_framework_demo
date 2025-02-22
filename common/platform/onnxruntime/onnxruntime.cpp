//
// Created by kaylor chen on 2024/6/9.
//

#include "onnxruntime.h"

#include "kaylordut/log/logger.h"

std::map<ONNXTensorElementDataType, size_t> datatype2size_map = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
};

void OnnxRuntime::Initialize(const char *model_path) {
  config_.model_name_path = model_path;
  config_.model_format = ModelFormat::ONNX_FORMAT;
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                    "OnnxRuntime_Instance");
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetInterOpNumThreads(1);
  // 如果有GPU支持，使用以下代码来开启
  //  OrtSessionOptionsAppendExecutionProvider_CUDA(*session_options_, 0);
  session_ =
      std::make_unique<Ort::Session>(*env_, model_path, *session_options_);

  // 打印模型输入层的信息（名称和类型）
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session_->GetInputCount();
  config_.input_tensors_count = num_input_nodes;

  for (std::size_t i = 0; i < num_input_nodes; ++i) {
    // 打印输入层的细节
    auto input_name = session_->GetInputNameAllocated(i, allocator);
    // 打印输入数据的维度和类型
    auto type_info = session_->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto data_type = tensor_info.GetElementType();
    input_tensors_data_type_.push_back(data_type);
    config_.input_single_element_size[std::string(input_name.get())] =
        datatype2size_map[data_type];
    config_.input_layer_shape[std::string(input_name.get())] =
        tensor_info.GetShape();
    config_.input_element_count[std::string(input_name.get())] =
        tensor_info.GetElementCount();
    config_.tensor_size.emplace(
        std::string(input_name.get()),
        tensor_info.GetElementCount() * datatype2size_map[data_type]);
    config_.input_index_to_name.emplace(i, std::string(input_name.get()));
  }
  size_t num_output_nodes = session_->GetOutputCount();
  config_.output_tensors_count = num_output_nodes;
  for (size_t i = 0; i < num_output_nodes; ++i) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);
    auto type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto data_type = tensor_info.GetElementType();
    output_tensors_data_type_.push_back(data_type);
    config_.output_single_element_size[std::string(output_name.get())] =
        datatype2size_map[data_type];
    config_.output_layer_shape[std::string(output_name.get())] =
        tensor_info.GetShape();
    config_.output_element_count[std::string(output_name.get())] =
        tensor_info.GetElementCount();
    config_.tensor_size.emplace(
        std::string(output_name.get()),
        tensor_info.GetElementCount() * datatype2size_map[data_type]);
    config_.output_index_to_name.emplace(i, std::string(output_name.get()));
  }
}

void OnnxRuntime::BindInputAndOutput(ai_framework::TensorData &tensor_data) {
  auto input = tensor_data.get_input_tensor_ptr();
  auto output = tensor_data.get_output_tensor_ptr();
  input_tensors_.clear();
  output_tensors_.clear();
  input_names_.clear();
  output_names_.clear();
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  size_t i = 0;
  for (const auto &kv : config_.input_element_count) {
    input_names_.push_back(kv.first.c_str());
    auto bytes = kv.second * config_.input_single_element_size[kv.first];
    auto &shape = config_.input_layer_shape[kv.first];
    auto &data_type = input_tensors_data_type_.at(i);
    input_tensors_.push_back(
        Ort::Value::CreateTensor(memory_info, (void *)input[i], bytes,
                                 shape.data(), shape.size(), data_type));
    i++;
  }
  i = 0;
  for (const auto &kv : config_.output_element_count) {
    output_names_.push_back(kv.first.c_str());
    auto bytes = kv.second * config_.output_single_element_size[kv.first];
    auto &shape = config_.output_layer_shape[kv.first];
    auto &data_type = output_tensors_data_type_.at(i);
    output_tensors_.push_back(
        Ort::Value::CreateTensor(memory_info, (void *)output[i], bytes,
                                 shape.data(), shape.size(), data_type));
    i++;
  }
}

void OnnxRuntime::DoInference() {
  Ort::RunOptions run_options;
  session_->Run(run_options, input_names_.data(), input_tensors_.data(),
                input_tensors_.size(), output_names_.data(),
                output_tensors_.data(), output_tensors_.size());
}