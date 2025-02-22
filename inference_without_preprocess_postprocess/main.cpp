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

int main(int argc, char *argv[]) {
  std::string model_path = argv[1];
#if defined(ONNX)
#define INSTANCE OnnxRuntime
#elif defined(TRT)
#define INSTANCE TensorRT
#elif defined(NNRT)
#define INSTANCE Nnrt
#elif defined(RK3588)
#define INSTANCE Rk3588
#endif
  auto instance = std::make_shared<INSTANCE>();
  instance->Initialize(model_path.c_str());
  auto tensor_data =
      std::make_shared<ai_framework::TensorData>(instance->get_config());
  instance->BindInputAndOutput(*tensor_data);
  auto input = tensor_data->get_input_tensor_ptr();
  auto output = tensor_data->get_output_tensor_ptr();
  auto input_data = static_cast<float *>(input[0]);
  auto output_data = static_cast<float *>(output[0]);
  for (int i = 0; i < 300; ++i) {
    input_data[i] = i;
  }
  KAYLORDUT_TIME_COST_INFO("Inference", instance->DoInference());
  for (int i = 0; i < 100; ++i) {
    KAYLORDUT_LOG_INFO("output_data[{}] = {}", i, output_data[i]);
  }
  return 0;
}
