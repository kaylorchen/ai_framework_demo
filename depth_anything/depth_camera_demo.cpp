//
// Created by kaylor on 7/10/24.
//
#include "common/ai_framework.h"
#include "image_process/depth_anything/depth_imageprocess.h"
#include "kaylordut/log/logger.h"
#ifdef TRT
#include "common/platform/tensorrt/tensorrt.h"
#endif
#ifdef ONNX
#include "common/platform/onnxruntime/onnxruntime.h"
#endif
#ifdef RK3588
#include "common/platform/rockchip/rk3588.h"
#endif
#ifdef NNRT
#include "common/platform/nnrt/nnrt.h"
#endif
#include "kaylordut/time/time.h"
#include "yaml-cpp/yaml.h"

int main(int argc, char **argv) {
  YAML::Node config = YAML::LoadFile("../config/depth_anything.yaml");
  auto model_path = config["model_path"].as<std::string>();
  int input_size = config["input_size"].as<int>();
  cv::VideoCapture capture(2, cv::CAP_V4L2);
  if (!capture.isOpened()) {
    KAYLORDUT_LOG_ERROR("Cannot open /dev/video{}", 0);
    exit(1);
  }
  capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 800);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
  capture.set(cv::CAP_PROP_FPS, 30);
#ifdef TRT
  auto ai_instance = std::make_shared<TensorRT>();
#elif defined ONNX
  auto ai_instance = std::make_shared<OnnxRuntime>();
#elif defined RK3588
  auto ai_instance = std::make_shared<Rk3588>(true);
#elif defined NNRT
  auto ai_instance = std::make_shared<Nnrt>();
#endif
  ai_instance->Initialize(model_path.c_str());
  ai_instance->PrintLayerInfo();
  ai_framework::TensorData tensor_data(ai_instance->get_config());
  DepthImageprocess depth_imageprocess(ai_instance->get_config(), input_size,
                                       {0.485, 0.456, 0.406},
                                       {0.229, 0.224, 0.225});
  ai_instance->BindInputAndOutput(tensor_data);
  std::vector<cv::Mat> input(1);
  int count = 300;
  for (int i = 0; i < count; ++i) {
    capture >> input.at(0);
    if (input.at(0).empty()) {
      KAYLORDUT_LOG_ERROR("Could not read image");
      continue;
    }
    depth_imageprocess.PreProcess(input, tensor_data.get_input_tensor_ptr());
    KAYLORDUT_TIME_COST_INFO("DoInference()", ai_instance->DoInference());
    depth_imageprocess.PostProcess(tensor_data.get_output_tensor_ptr());
    cv::Mat res;
    if (!depth_imageprocess.GetResult(res)) {
      KAYLORDUT_LOG_ERROR("No result");
    }
    KAYLORDUT_TIME_COST_INFO("Show result", {
      cv::Mat display;
      cv::hconcat(input.at(0), res, display);
      //    kaylordut::Time time;
      //    auto filename = time.now_str() + ".jpg";
      //    cv::imwrite(filename, display);
      cv::imshow("result", display);
      cv::waitKey(1);
    });
  }
  return 0;
}
