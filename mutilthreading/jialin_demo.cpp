//
// Created by kaylor chen on 2024/6/22.
//
#include "kaylordut/log/logger.h"
#include "kaylordut/time/run_once.h"
#include "kaylordut/time/time_duration.h"
#include "utils/tools.h"
#include "utils/videofile.h"
#include "utils/yolo_threadpool.h"
#include "yaml-cpp/yaml.h"

double get_now() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}

int main(int argc, char **argv) {
  std::string model_path = "/opt/jialin/model/yolo/yolo_seg.trt";
  std::string image_path = "../model/image/jialin.jpg";
  KAYLORDUT_LOG_INFO("model_path = {}", model_path);
  YAML::Node postprocess_config =
      YAML::LoadFile("../config/jialin_yolo_postprocess.yaml");
  std::vector<float> confidence_thresholds;
  std::vector<std::string> labels;
  for (const auto &kv : postprocess_config) {
    labels.push_back(kv.first.as<std::string>());
    confidence_thresholds.push_back(
        kv.second["confidence_threshold"].as<float>());
  }
  YoloThreadpool yolo_threadpool(model_path, confidence_thresholds, 1);
  std::vector<cv::Mat> input(1);
  input.at(0) = cv::imread(image_path);
  std::shared_ptr<YoloThreadpool::YoloInferenceResult> result = nullptr;
  if (!input.at(0).empty()) {
    std::vector<double> time;
    time.push_back(get_now());
    yolo_threadpool.AddInferenceTask(input, time, true);
  }
  {
    std::unique_lock<std::mutex> lock(yolo_threadpool.get_result_mutex());
    yolo_threadpool.get_result_condition_variable().wait(
        lock, [&] { return yolo_threadpool.get_result_ready(); });
  }
  KAYLORDUT_LOG_INFO("Inference finished");
  result = yolo_threadpool.GetInferenceResult();
  if (result != nullptr) {
    KAYLORDUT_TIME_COST_DEBUG(
        "ShowResults()",
        ShowResults(result->original_image.at(0),
                    yolo_threadpool.get_model_input_side_length(),
                    result->results, labels, 1, false, false));
  }
  cv::waitKey();
  cv::destroyAllWindows();
  return 0;
}
