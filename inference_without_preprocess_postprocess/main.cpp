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

  const int num_iterations = 1000;
  double total_time = 0.0;

  for (int i = 0; i < num_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    KAYLORDUT_TIME_COST_INFO("Inference", instance->DoInference(););
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    total_time += duration.count();
  }

  double average_time = total_time / num_iterations;

  KAYLORDUT_LOG_INFO("Inference loop {} times, average inference time: {:.3f} ms", num_iterations,
                     average_time);
  return 0;
}
