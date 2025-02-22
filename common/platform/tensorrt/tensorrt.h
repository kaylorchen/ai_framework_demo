//
// Created by kaylor on 6/15/24.
//

#pragma once
#include "NvInferPlugin.h"
#include "ai_framework.h"
#include "memory"

class Logger : public nvinfer1::ILogger {
 public:
  nvinfer1::ILogger::Severity reportableSeverity;
  explicit Logger(
      nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
      : reportableSeverity(severity) {}
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override;
};

class TensorRT : public ai_framework::AiInstance {
 public:
  TensorRT() = default;
  ~TensorRT();
  virtual void Initialize(const char *model_path) final;
  virtual void BindInputAndOutput(ai_framework::TensorData &tensor_data) final;
  virtual void DoInference() final;
  void set_async(bool async) { async_ = async; }
  bool get_async() { return async_; }

 private:
  std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr};
  std::shared_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  cudaStream_t stream_{nullptr};
  Logger gLogger_{nvinfer1::ILogger::Severity::kERROR};
  bool async_{true};
  ai_framework::TensorData *tensor_data_;
  std::vector<void *> device_ptrs;
};
