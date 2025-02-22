//
// Created by kaylor chen on 2024/6/9.
//

#pragma once
#include <onnxruntime_cxx_api.h>

#include "ai_framework.h"
#include "memory"

class OnnxRuntime : public ai_framework::AiInstance {
 public:
  virtual void Initialize(const char *model_path) final;
  virtual void BindInputAndOutput(ai_framework::TensorData &tensor_data) final;
  virtual void DoInference() final;

 private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::vector<Ort::Value> input_tensors_;
  std::vector<ONNXTensorElementDataType> input_tensors_data_type_;
  std::vector<Ort::Value> output_tensors_;
  std::vector<ONNXTensorElementDataType> output_tensors_data_type_;
  std::vector<const char *> input_names_;
  std::vector<const char *> output_names_;
};
