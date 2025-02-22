//
// Created by kaylor on 6/20/24.
//

#ifndef AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_
#define AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_

#include "ai_framework.h"
#include "rknn_api.h"

class Rk3588 : public ai_framework::AiInstance {
 public:
  Rk3588(bool zero_copy = true);
  Rk3588(rknn_context *ctx_in, bool zero_copy = true);
  ~Rk3588();
  virtual void Initialize(const char *model_path) final;
  virtual void BindInputAndOutput(ai_framework::TensorData &tensor_data) final;
  virtual void DoInference() final;
  rknn_context *get_context() { return &ctx_; }

 private:
  rknn_context ctx_{};
  rknn_context *dup_ctx_{nullptr};
  rknn_core_mask core_mask_;
  static uint16_t instance_count_;
  rknn_input *input_{nullptr};
  rknn_output *output_{nullptr};
  rknn_tensor_attr *input_attr_{nullptr};
  rknn_tensor_attr *output_attr_{nullptr};
  bool zero_copy_{true};
};

#endif  // AI_FRAMEWORK_PLATFORM_ROCKCHIP_RK3588_H_
