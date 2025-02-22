//
// Created by kaylor on 6/15/24.
//

#include "tensorrt.h"

#include <cassert>
#include <fstream>

#include "kaylordut/log/logger.h"
#include "thread"

std::map<nvinfer1::DataType, size_t> datatype2size = {
    {nvinfer1::DataType::kFLOAT, sizeof(float)},
    {nvinfer1::DataType::kHALF, sizeof(uint16_t)},
    {nvinfer1::DataType::kINT8, sizeof(uint8_t)},
    {nvinfer1::DataType::kINT32, sizeof(int32_t)},
};

void TensorRT::Initialize(const char *model_path) {
  config_.model_name_path = model_path;
  config_.model_format = ModelFormat::TRT_FORMAT;
  std::ifstream file(model_path, std::ios::binary);
  assert(file.good());
  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *trt_model_stream = new char[size];
  assert(trt_model_stream);
  file.read(trt_model_stream, size);
  file.close();

  initLibNvInferPlugins(&this->gLogger_, "");
  runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(gLogger_));
  assert(runtime_ != nullptr);
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(trt_model_stream, size));
  assert(engine_ != nullptr);
  delete[] trt_model_stream;
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  assert(context_ != nullptr);
  KAYLORDUT_LOG_INFO("n = {} ", engine_->getNbIOTensors());
  auto num = engine_->getNbIOTensors();
  int input_index = 0;
  int output_index = 0;
  for (int i = 0; i < num; ++i) {
    auto name = engine_->getIOTensorName(i);
    auto io = engine_->getTensorIOMode(name);
    if (io == nvinfer1::TensorIOMode::kINPUT) {
      config_.input_index_to_name.emplace(config_.input_tensors_count, name);
      config_.input_tensors_count++;
      auto dim = engine_->getTensorShape(name);
      std::vector<int64_t> shape;
      size_t count = 1;
      for (int j = 0; j < dim.nbDims; ++j) {
        shape.push_back(dim.d[j]);
        count *= dim.d[j];
      }
      config_.input_layer_shape.emplace(name, shape);
      config_.input_element_count.emplace(name, count);
      auto data_type = engine_->getTensorDataType(name);
      config_.input_single_element_size.emplace(name,
                                                datatype2size.at(data_type));
      config_.tensor_size.emplace(name, datatype2size.at(data_type) * count);
      config_.input_index_to_name.emplace(input_index, name);
      ++input_index;
    } else if (io == nvinfer1::TensorIOMode::kOUTPUT) {
      config_.input_index_to_name.emplace(config_.input_tensors_count, name);
      config_.output_tensors_count++;
      auto dim = engine_->getTensorShape(name);
      std::vector<int64_t> shape;
      size_t count = 1;
      for (int j = 0; j < dim.nbDims; ++j) {
        shape.push_back(dim.d[j]);
        count *= dim.d[j];
      }
      config_.output_layer_shape.emplace(name, shape);
      config_.output_element_count.emplace(name, count);
      auto data_type = engine_->getTensorDataType(name);
      config_.output_single_element_size.emplace(name,
                                                 datatype2size.at(data_type));
      config_.tensor_size.emplace(name, datatype2size.at(data_type) * count);
      config_.output_index_to_name.emplace(output_index, name);
      ++output_index;
    }
  }
  //  context_->setInputTensorAddress()
  //  context_->setTensorAddress()
}

void TensorRT::BindInputAndOutput(ai_framework::TensorData &tensor_data) {
  this->tensor_data_ = &tensor_data;
  auto input = this->tensor_data_->get_input_tensor_cuda_ptr();
  auto output = this->tensor_data_->get_output_tensor_cuda_ptr();
  auto input_name = this->tensor_data_->get_input_tensor_name();
  auto output_name = this->tensor_data_->get_output_tensor_name();
  device_ptrs.clear();
  for (int i = 0; i < input_name.size(); ++i) {
    context_->setInputTensorAddress(input_name.at(i).c_str(), input[i]);
    device_ptrs.push_back(input[i]);
  }
  for (int i = 0; i < output_name.size(); ++i) {
    context_->setTensorAddress(output_name.at(i).c_str(), output[i]);
    device_ptrs.push_back(output[i]);
  }
}

void TensorRT::DoInference() {
  auto input = this->tensor_data_->get_input_tensor_ptr();
  auto output = this->tensor_data_->get_output_tensor_ptr();
  auto input_cuda = this->tensor_data_->get_input_tensor_cuda_ptr();
  auto output_cuda = this->tensor_data_->get_output_tensor_cuda_ptr();
  auto input_size = this->tensor_data_->get_input_tensor_size();
  auto output_size = this->tensor_data_->get_output_tensor_size();
  if (async_ == false) {
    // 从host拷贝到device
    for (int i = 0; i < this->get_input_tensor_count(); ++i) {
      auto status = cudaMemcpy(input_cuda[i], input[i], input_size.at(i),
                               cudaMemcpyHostToDevice);
      KAYLORDUT_LOG_ERROR_EXPRESSION(status != cudaSuccess,
                                     "cudaMemcpy failed, status: {}", status);
    }
    bool status = context_->executeV2(device_ptrs.data());
    KAYLORDUT_LOG_ERROR_EXPRESSION(status == false, "Inference failed");
    for (int i = 0; i < this->get_output_tensor_count(); ++i) {
      auto status = cudaMemcpy(output[i], output_cuda[i], output_size.at(i),
                               cudaMemcpyDeviceToHost);
      KAYLORDUT_LOG_ERROR_EXPRESSION(status != cudaSuccess,
                                     "cudaMemcpy failed, status: {}", status);
    }
  } else {
    if (stream_ == nullptr) {
      cudaStreamCreate(&stream_);
    }
    // 从host拷贝到device
    for (int i = 0; i < this->get_input_tensor_count(); ++i) {
      auto status = cudaMemcpyAsync(input_cuda[i], input[i], input_size.at(i),
                                    cudaMemcpyHostToDevice, stream_);
      KAYLORDUT_LOG_ERROR_EXPRESSION(
          status != cudaSuccess, "cudaMemcpyAsync failed, status: {}", status);
    }
    bool status = context_->enqueueV3(stream_);
    KAYLORDUT_LOG_ERROR_EXPRESSION(status == false, "Inference failed");
    for (int i = 0; i < this->get_output_tensor_count(); ++i) {
      auto status =
          cudaMemcpyAsync(output[i], output_cuda[i], output_size.at(i),
                          cudaMemcpyDeviceToHost, stream_);
      KAYLORDUT_LOG_ERROR_EXPRESSION(
          status != cudaSuccess, "cudaMemcpyAsync failed, status: {}", status);
    }
    auto sync_status = cudaStreamSynchronize(stream_);
    KAYLORDUT_LOG_ERROR_EXPRESSION(sync_status != cudaSuccess,
                                   "cudaStreamSynchronize failed, status: {}",
                                   sync_status);
  }
}

TensorRT::~TensorRT() {
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (stream_ != nullptr) {
    cudaStreamDestroy(this->stream_);
  }
}

void Logger::log(nvinfer1::ILogger::Severity severity,
                 const char *msg) noexcept {
  if (severity > reportableSeverity) {
    return;
  }
  KAYLORDUT_LOG_ERROR("{}", msg);
}
