//
// Created by kaylor on 7/11/24.
//

#ifndef AI_FRAMEWORK_COMMON_AI_INSTANCE_H_
#define AI_FRAMEWORK_COMMON_AI_INSTANCE_H_
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#ifdef RK3588
#include "rknn_api.h"
#endif
enum ModelFormat : uint8_t {
  ONNX_FORMAT = 0,
  TRT_FORMAT,
  RKNN_FORMAT,
  NNRT_FORMAT,
  UNKNOWN_FORMAT,
};
namespace ai_framework {
struct Config {
  ModelFormat model_format{UNKNOWN_FORMAT};
  std::string model_name_path;
  uint16_t input_tensors_count{0};
  uint16_t output_tensors_count{0};
  std::map<uint16_t, std::string> input_index_to_name;
  std::map<uint16_t, std::string> output_index_to_name;
  std::map<std::string, size_t> input_element_count;
  std::map<std::string, size_t> output_element_count;
  std::map<std::string, size_t> input_single_element_size;
  std::map<std::string, size_t> output_single_element_size;
  std::map<std::string, std::vector<int64_t>> input_layer_shape;
  std::map<std::string, std::vector<int64_t>> output_layer_shape;
  std::map<std::string, float> scale;
  std::map<std::string, int> zero_point;
  std::map<std::string, size_t> tensor_size;
#ifdef RK3588
  rknn_context rknn_ctx;
  bool rknn_zero_copy{true};
  std::map<std::string, bool> width_equal_stride;
  std::map<std::string, uint32_t> stride;
#endif
#ifdef NNRT
  bool is_device{true};
#endif
};
class TensorData;
using TensorDataPtr = std::shared_ptr<TensorData>;
class AiInstance;
using AiInstancePtr = std::shared_ptr<AiInstance>;
class Engine {
 public:
  using Ptr = std::shared_ptr<Engine>;
  Engine() = delete;
  Engine(const enum ModelFormat format, const char *model_path);
  void **&get_input_tensor_ptr();
  const void *const *get_output_tensor_ptr();
  const int get_input_tensor_count();
  const int get_output_tensor_count();
  const std::map<std::string, std::vector<int64_t>> &get_input_tensor_shape()
      const;
  const std::map<std::string, std::vector<int64_t>> &get_output_tensor_shape()
      const;
  const std::map<std::string, float> &get_tensor_scale() const;
  const std::map<std::string, int> &get_tensor_zero_point() const;
  const ModelFormat get_model_format() const;
#ifdef RK3588
  const std::map<std::string, bool> &get_width_equal_stride() const;
  const std::map<std::string, uint32_t> &get_stride() const;
#endif
  void DoInference(void);

 protected:
  AiInstancePtr instance_ptr_;
  TensorDataPtr tensor_data_ptr_;
};
}  // namespace ai_framework

#endif  // AI_FRAMEWORK_COMMON_AI_INSTANCE_H_
