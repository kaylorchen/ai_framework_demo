//
// Created by kaylor on 12/1/25.
//
#include "kaylordut/log/logger.h"
#include "rclcpp/rclcpp.hpp"
#if defined(TRT)
#include "platform/tensorrt/tensorrt.h"
#elif defined(ONNX)
#include "platform/onnxruntime/onnxruntime.h"
#elif defined(RK3588)
#include "platform/rockchip/rk3588.h"
#elif defined(NNRT)
#include "platform/nnrt/nnrt.h"
#endif
#include "foundation_stereo_image_process.h"

class FoundationStereoNode : public rclcpp::Node {
public:
  FoundationStereoNode() : Node("foundation_stereo") {
    baseline_ = 0.1198430;
    K_ = {955.8550, 0.0, 655.0450, 0.0, 955.9950, 363.3060, 0.0, 0.0, 1.0};
    std::stringstream ss;
    ss << std::endl << "baseline = " << baseline_ << std::endl;
    ss << "K = " << std::endl;
    for (int i = 0; i < 3; ++i) {
      ss << "[ ";
      for (int j = 0; j < 3; ++j) {
        ss << std::setw(20) << std::setprecision(6) << std::fixed
           << K_[i * 3 + j];
      }
      ss << " ]" << std::endl;
    }
    KAYLORDUT_LOG_INFO("{}", ss.str());
    KAYLORDUT_LOG_INFO("Create a Ai Instance")
#if defined(TRT)
    ai_instance_ = std::make_shared<TensorRT>();
#elif defined(ONNX)
    auto ai_instance = std::make_shared<OnnxRuntime>();
#elif defined(RK3588)
    auto ai_instance = std::make_shared<Rk3588>(true);
#elif defined(NNRT)
    auto ai_instance = std::make_shared<Nnrt>();
#endif
    ai_instance_->Initialize(model_path_.c_str());
    ai_instance_->PrintLayerInfo();
    tensor_data_ =
        std::make_shared<ai_framework::TensorData>(ai_instance_->get_config());
    ai_instance_->BindInputAndOutput(*tensor_data_);
    image_process_ = std::make_shared<FoundationStereoImageProcess>(
        ai_instance_->get_config(), K_, baseline_);
    cap_ = cv::VideoCapture(
        "v4l2src device=" + device_ + " ! video/x-raw,format=YUY2,width=" +
            std::to_string(width_) + ",height=" + std::to_string(height_) +
            ",framerate=" + std::to_string(fps_) +
            "/1 ! videoconvert ! appsink",
        cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
      KAYLORDUT_LOG_ERROR("Failed to open camera");
      exit(-1);
    }
    thread_ = std::thread([&] { this->Run(); });
  }
  ~FoundationStereoNode() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  float baseline_;
  std::vector<float> K_;
#if defined(TRT)
  std::shared_ptr<TensorRT> ai_instance_;
#elif defined(ONNX)
  std::shared_ptr<OnnxRuntime> ai_instance_;
#elif defined(RK3588)
  std::shared_ptr<Rk3588> ai_instance_;
#elif defined(NNRT)
  std::shared_ptr<Nnrt> ai_instance_;
#endif
  int width_{2560};
  int height_{720};
  int fps_{60};
  std::string device_{"/dev/video0"};
  std::string model_path_{
      "/home/kaylor/work/kaylor/nvidia/FoundationStereo/pretrained_models/"
      "480x288_small/foundation_stereo_10.13.2.6_fp16.trt"};
  std::shared_ptr<ai_framework::TensorData> tensor_data_;
  std::shared_ptr<FoundationStereoImageProcess> image_process_;
  std::shared_ptr<FoundationStereoImageProcess::PreProcessResult>
      pre_process_result_;
  std::shared_ptr<FoundationStereoImageProcess::PostProcessResult>
      post_process_result_;
  cv::VideoCapture cap_;
  std::thread thread_;
  void Run() {
    std::vector<cv::Mat> imgs(2);
    cv::Mat frame;
    while (rclcpp::ok()) {
      if (!cap_.read(frame)) {
        KAYLORDUT_LOG_ERROR("Failed to read frame")
        exit(-1);
      }
      imgs[0] = frame(cv::Rect(0, 0, width_ / 2, height_));
      imgs[1] = frame(cv::Rect(width_ / 2, 0, width_ / 2, height_));
      KAYLORDUT_TIME_COST_INFO("PreProcess()",
                               pre_process_result_ = image_process_->PreProcess(
                                   imgs, tensor_data_->get_input_tensor_ptr()));
      KAYLORDUT_TIME_COST_INFO("DoInference()", ai_instance_->DoInference());
      KAYLORDUT_TIME_COST_INFO(
          "PostProcess()",
          post_process_result_ = image_process_->PostProcess(
              tensor_data_->get_output_tensor_ptr(), *pre_process_result_));
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FoundationStereoNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
