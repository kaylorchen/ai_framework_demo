//
// Created by kaylor on 7/11/24.
//

#include "ai_instance.h"

#include "ai_framework.h"
#ifdef TRT
#include "platform/tensorrt/tensorrt.h"
#endif
#ifdef ONNX
#include "platform/onnxruntime/onnxruntime.h"
#endif
#ifdef RK3588
#include "platform/rockchip/rk3588.h"
#endif
#ifdef NNRT
#include "platform/nnrt/nnrt.h"
#endif

ai_framework::Engine::Engine(const enum ModelFormat format,
                             const char *model_path) {
  switch (format) {
#ifdef TRT
    case ModelFormat::TRT_FORMAT:
      instance_ptr_ = std::make_shared<TensorRT>();
      break;
#endif
#ifdef ONNX
    case ModelFormat::ONNX_FORMAT:
      instance_ptr_ = std::make_shared<OnnxRuntime>();
      break;
#endif
#ifdef RK3588
    case ModelFormat::RKNN_FORMAT:
      instance_ptr_ = std::make_shared<Rk3588>();
      break;
#endif
#ifdef NNRT
    case ModelFormat::NNRT_FORMAT:
      instance_ptr_ = std::make_shared<Nnrt>();
      break;
#endif
    default:
      perror("Unknown format\n");
      exit(EXIT_FAILURE);
  }
  instance_ptr_->Initialize(model_path);
  instance_ptr_->PrintLayerInfo();
  tensor_data_ptr_ = std::make_shared<TensorData>(instance_ptr_->get_config());
  instance_ptr_->BindInputAndOutput(*tensor_data_ptr_);
}

void ai_framework::Engine::DoInference() { instance_ptr_->DoInference(); }

void **&ai_framework::Engine::get_input_tensor_ptr() {
  return tensor_data_ptr_->get_input_tensor_ptr();
}

const void *const *ai_framework::Engine::get_output_tensor_ptr() {
  return tensor_data_ptr_->get_output_tensor_ptr();
}

const std::map<std::string, std::vector<int64_t>>
    &ai_framework::Engine::get_input_tensor_shape() const {
  return instance_ptr_->get_config().input_layer_shape;
}

const std::map<std::string, std::vector<int64_t>>
    &ai_framework::Engine::get_output_tensor_shape() const {
  return instance_ptr_->get_config().output_layer_shape;
}

const std::map<std::string, float> &ai_framework::Engine::get_tensor_scale()
    const {
  return instance_ptr_->get_config().scale;
}

const std::map<std::string, int> &ai_framework::Engine::get_tensor_zero_point()
    const {
  return instance_ptr_->get_config().zero_point;
}

const int ai_framework::Engine::get_input_tensor_count() {
  return tensor_data_ptr_->get_input_tensor_count();
}

const int ai_framework::Engine::get_output_tensor_count() {
  return tensor_data_ptr_->get_output_tensor_count();
}

const ModelFormat ai_framework::Engine::get_model_format() const {
  return instance_ptr_->get_config().model_format;
}

#ifdef RK3588
const std::map<std::string, bool>
    &ai_framework::Engine::get_width_equal_stride() const {
  return instance_ptr_->get_config().width_equal_stride;
}

const std::map<std::string, uint32_t> &ai_framework::Engine::get_stride()
    const {
  return instance_ptr_->get_config().stride;
}
#endif
