//
// Created by kaylor on 11/25/25.
//
#include "kaylordut/log/logger.h"

#include <filesystem>
#if defined(TRT)
#include "platform/tensorrt/tensorrt.h"
#elif defined(ONNX)
#include "platform/onnxruntime/onnxruntime.h"
#elif defined(RK3588)
#include "platform/rockchip/rk3588.h"
#elif defined(NNRT)
#include "platform/nnrt/nnrt.h"
#endif
#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "stereo_image_process.h"
#include "yaml-cpp/yaml.h"
int main(int argc, char **argv) {
  KAYLORDUT_LOG_INFO("foundation_stereo demo");
  auto config = YAML::LoadFile("../config/stereo.yaml");
  std::string model_path = config["model_path"].as<std::string>();
  auto left_image_path = config["image_path"]["left"].as<std::string>();
  auto right_image_path = config["image_path"]["right"].as<std::string>();
  KAYLORDUT_LOG_INFO("left_image_path = {}", left_image_path);
  KAYLORDUT_LOG_INFO("right_image_path = {}", right_image_path);
  auto baseline = config["baseline"].as<float>();
  auto K = config["K"].as<std::vector<float>>();
  auto doffs = config["doffs"].as<float>();
  std::stringstream ss;
  ss << std::endl
     << "baseline = " << baseline << " doffs = " << doffs << std::endl;
  ss << "K = " << std::endl;
  for (int i = 0; i < 3; ++i) {
    ss << "[ ";
    for (int j = 0; j < 3; ++j) {
      ss << std::setw(20) << std::setprecision(6) << std::fixed << K[i * 3 + j];
    }
    ss << " ]" << std::endl;
  }
  KAYLORDUT_LOG_INFO("{}", ss.str());

  auto left_image = cv::imread(left_image_path);
  auto right_image = cv::imread(right_image_path);
  // cv::imshow("left_image", left_image);
  // cv::imshow("right_image", right_image);
  // cv::waitKey();
#if defined(TRT)
  auto ai_instance = std::make_shared<TensorRT>();
#elif defined(ONNX)
  auto ai_instance = std::make_shared<OnnxRuntime>();
#elif defined(RK3588)
  auto ai_instance = std::make_shared<Rk3588>(true);
#elif defined(NNRT)
  auto ai_instance = std::make_shared<Nnrt>();
#endif
  ai_instance->Initialize(model_path.c_str());
  ai_instance->PrintLayerInfo();
  ai_framework::TensorData tensor_data(ai_instance->get_config());
  std::vector<cv::Mat> imgs = {left_image, right_image};
  FoundationStereoImageProcess image_process(ai_instance->get_config(), K,
                                             baseline, doffs);
  auto pre_process_result =
      image_process.PreProcess(imgs, tensor_data.get_input_tensor_ptr());
  ai_instance->BindInputAndOutput(tensor_data);
  for (size_t i = 0; i < 1; ++i) {
    KAYLORDUT_TIME_COST_INFO("DoInference()", ai_instance->DoInference());
  }
  auto res = image_process.PostProcess(tensor_data.get_output_tensor_ptr(),
                                       *pre_process_result);
  std::string filename = std::filesystem::path(model_path).stem().string();
  pcl::io::savePCDFile(filename + ".pcd", *res->cloud);
  pcl::io::savePLYFile(filename + ".ply", *res->cloud);
  // cv::imshow("depth_map", res->depth_img);
  // cv::waitKey(5000);
  return 0;
}
