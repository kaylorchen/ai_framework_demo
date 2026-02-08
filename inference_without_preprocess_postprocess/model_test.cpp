//
// Created by ubuntu on 2/7/25.
//
#include "ai_instance.h"
#include "kaylordut/log/logger.h"
#ifdef RK3588
#include "platform/rockchip/rk3588.h"
#endif
#ifdef ONNX
#include "platform/onnxruntime/onnxruntime.h"
#endif
#ifdef TRT
#include "platform/tensorrt/tensorrt.h"
#endif
#ifdef NNRT
#include "platform/nnrt/nnrt.h"
#endif

#ifdef __aarch64__
#include "arm_neon.h"
#endif


int main(int argc, char *argv[]) {
  std::string model_path = argv[1];
#if defined(ONNX)
#define INSTANCE OnnxRuntime
  auto instance = std::make_shared<INSTANCE>();
#elif defined(TRT)
#define INSTANCE TensorRT
  auto instance = std::make_shared<INSTANCE>();
#elif defined(NNRT)
#define INSTANCE Nnrt
  auto instance = std::make_shared<INSTANCE>();
#elif defined(RK3588)
#define INSTANCE Rk3588
  auto instance = std::make_shared<INSTANCE>(true, RKNN_TENSOR_FLOAT16);
#endif
  instance->Initialize(model_path.c_str());
  auto tensor_data =
      std::make_shared<ai_framework::TensorData>(instance->get_config());
  instance->BindInputAndOutput(*tensor_data);
  instance->PrintLayerInfo();
  auto input = tensor_data->get_input_tensor_ptr();
  auto output = tensor_data->get_output_tensor_ptr();

  float input_values[] = {-0.00, 0.00, 0.01, 1.00, -0.00, 0.03, 0.50, 0.00, 0.00, 0.44, 0.23, 8.78, 4.69};
#ifndef __aarch64__
  float *input_data = static_cast<float *>(input[0]);
  float *output_data = static_cast<float *>(output[0]);
#else
  float16_t *input_data = static_cast<float16_t *>(input[0]);
  float16_t *output_data = static_cast<float16_t *>(output[0]);
#endif
  // 使用给定的输入数据
  for (int i = 0; i < 13; ++i) {
    input_data[i] = input_values[i];
    KAYLORDUT_LOG_INFO("input_data[{}] = {}", i, static_cast<float>(input_data[i]));
  }
  KAYLORDUT_TIME_COST_INFO("Inference", instance->DoInference(););
  for (int i = 0; i < 2; ++i) {
    KAYLORDUT_LOG_INFO("output_data[{}]: {}", i, static_cast<float>(output_data[i]));
  }
  return 0;
}
