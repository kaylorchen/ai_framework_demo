//
// Created by kaylor on 7/10/24.
//
#include "common/ai_framework.h"
#include "image_process/depth_anything/depth_imageprocess.h"
#include "kaylordut/log/logger.h"
#ifdef TRT
#include "platform/tensorrt/tensorrt.h"
#endif
#ifdef ONNX
#include "platform/onnxruntime/onnxruntime.h"
#endif
#ifdef RK3588
#include "platform/rockchip/rk3588.h"
#endif
#include "kaylordut/time/time.h"
#include "yaml-cpp/yaml.h"

int main(int argc, char **argv) {
  YAML::Node config = YAML::LoadFile("../config/depth_anything.yaml");
  auto model_path = config["model_path"].as<std::string>();
  auto image_path = config["image_path"].as<std::string>();
  int input_size = config["input_size"].as<int>();
#ifdef TRT
  auto ai_instance = std::make_shared<TensorRT>();
#elif ONNX
  auto ai_instance = std::make_shared<OnnxRuntime>();
#elif RK3588
  auto ai_instance = std::make_shared<Rk3588>(true);
#endif
  ai_instance->Initialize(model_path.c_str());
  ai_instance->PrintLayerInfo();
  ai_framework::TensorData tensor_data(ai_instance->get_config());
  DepthImageprocess depth_imageprocess(ai_instance->get_config(), input_size,
                                       {0.485, 0.456, 0.406},
                                       {0.229, 0.224, 0.225});
  std::vector<cv::Mat> input(1);
  input.at(0) = cv::imread(image_path);
  if (input.at(0).empty()) {
    KAYLORDUT_LOG_ERROR("Could not read image");
    return -1;
  }
  KAYLORDUT_LOG_INFO("Begin...");
  depth_imageprocess.PreProcess(input, tensor_data.get_input_tensor_ptr());
  ai_instance->BindInputAndOutput(tensor_data);
  KAYLORDUT_TIME_COST_INFO("DoInference()", ai_instance->DoInference());
  depth_imageprocess.PostProcess(tensor_data.get_output_tensor_ptr());
  cv::Mat res;
  if (!depth_imageprocess.GetResult(res)) {
    KAYLORDUT_LOG_ERROR("No result");
  }
  cv::Mat display;
  cv::hconcat(input.at(0), res, display);
  kaylordut::Time time;
  auto filename = time.now_str() + ".jpg";
  cv::imwrite(filename, display);
  cv::imshow("result", display);
  cv::waitKey(0);
  return 0;
}
