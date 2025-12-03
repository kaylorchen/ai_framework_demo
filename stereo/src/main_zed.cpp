//
// Created by kaylor on 11/25/25.
//
#include "kaylordut/log/logger.h"
#if defined(TRT)
#include "platform/tensorrt/tensorrt.h"
#elif defined(ONNX)
#include "platform/onnxruntime/onnxruntime.h"
#elif defined(RK3588)
#include "platform/rockchip/rk3588.h"
#elif defined(NNRT)
#include "platform/nnrt/nnrt.h"
#endif
#include "display_cloud.h"
#include "foundation_stereo_image_process.h"
#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "yaml-cpp/yaml.h"

int main(int argc, char **argv) {
  KAYLORDUT_LOG_INFO("foundation_stereo demo");
  auto config = YAML::LoadFile("../config/foundation_stereo_zed.yaml");
  std::string model_path = config["model_path"].as<std::string>();
  auto baseline = config["baseline"].as<float>();
  auto K = config["K"].as<std::vector<float>>();
  std::stringstream ss;
  ss << std::endl << "baseline = " << baseline << std::endl;
  ss << "K = " << std::endl;
  for (int i = 0; i < 3; ++i) {
    ss << "[ ";
    for (int j = 0; j < 3; ++j) {
      ss << std::setw(20) << std::setprecision(6) << std::fixed << K[i * 3 + j];
    }
    ss << " ]" << std::endl;
  }
  KAYLORDUT_LOG_INFO("{}", ss.str());
  int width = config["width"].as<int>(2560);
  int height = config["height"].as<int>(720);
  int fps = config["fps"].as<int>(60);
  std::string device = config["device"].as<std::string>("/dev/video0");
  KAYLORDUT_LOG_INFO("device = {},fps = {}, width = {}, height = {}", device,
                     fps, width, height);
  auto capture = cv::VideoCapture(
      "v4l2src device=" + device + " ! video/x-raw,format=YUY2,width=" +
          std::to_string(width) + ",height=" + std::to_string(height) +
          ",framerate=" + std::to_string(fps) + "/1 ! videoconvert ! appsink",
      cv::CAP_GSTREAMER);
  if (!capture.isOpened()) {
    KAYLORDUT_LOG_ERROR("Failed to open camera");
    return -1;
  }
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
  ai_instance->BindInputAndOutput(tensor_data);
  FoundationStereoImageProcess image_process(ai_instance->get_config(), K,
                                             baseline);
  std::shared_ptr<FoundationStereoImageProcess::PreProcessResult>
      pre_process_result;
  std::vector<cv::Mat> imgs(2);
  cv::Mat frame;
  DisplayCloud display_cloud;
  std::shared_ptr<FoundationStereoImageProcess::PostProcessResult>
      post_process_result;
  while (true) {
    if (!capture.read(frame)) {
      KAYLORDUT_LOG_ERROR("Failed to read frame");
      break;
    }
    imgs[0] = frame(cv::Rect(0, 0, width / 2, height));
    imgs[1] = frame(cv::Rect(width / 2, 0, width / 2, height));
    KAYLORDUT_TIME_COST_INFO("PreProcess()",
                             pre_process_result = image_process.PreProcess(
                                 imgs, tensor_data.get_input_tensor_ptr()));
    KAYLORDUT_TIME_COST_INFO("DoInference()", ai_instance->DoInference());
    KAYLORDUT_TIME_COST_INFO(
        "PostProcess()",
        post_process_result = image_process.PostProcess(
            tensor_data.get_output_tensor_ptr(), *pre_process_result));
    display_cloud.ShowCloud(post_process_result->cloud);
    // cv::imshow("Depth Map", post_process_result->depth_img);
    // if (cv::waitKey(1) == 'q') {
    //   break;
    // }
  }
  return 0;
}
