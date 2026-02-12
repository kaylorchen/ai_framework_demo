//
// Created by ubuntu on 2/7/25.
//
#include "ai_instance.h"
#include "kaylordut/log/logger.h"
#include "thread"

#include <csignal>
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

static std::atomic<bool> keep_running(true);

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
  instance->PrintLayerInfo();
  auto first_tensor_name = instance->get_config().input_index_to_name.at(0);
  auto first_tensor_size = instance->get_config().tensor_size.at(first_tensor_name);
  KAYLORDUT_LOG_INFO("First tensor name: {}, size: {}", first_tensor_name, first_tensor_size)
  auto input = tensor_data->get_input_tensor_ptr();
  auto output = tensor_data->get_output_tensor_ptr();
  uint8_t *input_data = static_cast<uint8_t *>(input[0]);



  std::signal(SIGINT, [](int) {
    keep_running = false;
  });

  int num_iterations = 0;
  double total_time = 0.0;

  KAYLORDUT_LOG_INFO("Starting inference loop. Press Ctrl+C to stop...");

  std::srand(std::time(nullptr));
  while (keep_running) {
    uint8_t tmp = std::rand() % 256;
    std::memset(input_data, tmp, first_tensor_size);
    auto start = std::chrono::high_resolution_clock::now();
    // KAYLORDUT_TIME_COST_INFO("Inference", instance->DoInference(););
    instance->DoInference();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    total_time += duration.count();
    num_iterations++;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  double average_time = num_iterations > 0 ? total_time / num_iterations : 0.0;

  KAYLORDUT_LOG_INFO("\nInference loop {} times, average inference time: {:.3f} ms", num_iterations,
                     average_time);
  return 0;
}
