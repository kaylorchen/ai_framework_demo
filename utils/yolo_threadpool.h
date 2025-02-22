//
// Created by kaylor chen on 2024/6/22.
//

#pragma once
#include "ai_instance.h"
#include "image_process/yolo/yolo_postprocess.h"
#include "image_process/yolo/yolo_preprocess.h"
#include "threadpool.h"

class YoloThreadpool {
 public:
  struct YoloInferenceResult {
    double time_stamp;
    std::vector<cv::Mat> original_image;
    std::vector<YoloPostProcess::Result> results;
  };
  YoloThreadpool(std::string &model_path, std::vector<float> &conf_threshold,
                 int threads = 1);

  void AddInferenceTask(const std::vector<cv::Mat> &original_image,
                        const double time_stamp,
                        const bool clone_original_image = true);

  std::shared_ptr<YoloThreadpool::YoloInferenceResult> GetInferenceResult();

  int get_task_size() const { return this->pool_->TasksSize(); }

  const int &get_model_input_side_length() const {
    return yolo_preprocess_.at(0)->get_target_side_length();
  }

 private:
  template <class T>
  void CreateAiInstance(std::string &model_path, int &threads) {
    assert(threads > 0);
    auto instance = std::make_shared<T>();
    instances_.push_back(instance);
    instances_.at(0)->Initialize(model_path.c_str());
    for (int i = 1; i < threads; ++i) {
#ifdef RK3588
      instances_.push_back(std::make_shared<T>(instance->get_context()));
#else
      instances_.push_back(std::make_shared<T>());
#endif
      instances_.at(i)->Initialize(model_path.c_str());
    }
  }

  int get_thread_id();

  int num_threads_{1};
  std::unique_ptr<std::mutex[]> threads_mutex_;
  std::unique_ptr<ThreadPool> pool_{nullptr};
  std::vector<std::shared_ptr<ai_framework::AiInstance>> instances_;
  std::vector<std::shared_ptr<ai_framework::TensorData>> tensors_data_;
  std::vector<std::shared_ptr<YoloPreProcess>> yolo_preprocess_;
  std::vector<std::shared_ptr<YoloPostProcess>> yolo_postprocess_;
  std::queue<std::shared_ptr<YoloInferenceResult>> yolo_inference_result_queue_;
  std::mutex result_mutex_;
  uint16_t id_{0};
  std::mutex id_mutex_;
};
