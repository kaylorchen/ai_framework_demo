//
// Created by kaylor chen on 2024/6/22.
//

#include "yolo_threadpool.h"

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
#include "fstream"
#include "tools.h"

void YoloThreadpool::Initialize(const char *model_data,
                                const uint64_t model_size,
                                const std::string model_extension,
                                std::vector<float> &conf_threshold,
                                int threads) {
  try {
    this->num_threads_ = threads;
    this->threads_mutex_ = std::make_unique<std::mutex[]>(threads);
    this->pool_ = std::make_unique<ThreadPool>(threads);
#ifdef RK3588
    if (model_extension == "rknn") {
      CreateAiInstance<Rk3588>(model_data, model_size, threads);
    }
#endif
#ifdef ONNX
    if (model_extension == "onnx") {
      CreateAiInstance<OnnxRuntime>(model_data, model_size, threads);
    }
#endif
#ifdef TRT
    if (model_extension == "trt" || model_extension == "engine") {
      CreateAiInstance<TensorRT>(model_data, model_size, threads);
    }
#endif
#ifdef NNRT
    if (model_extension == "om") {
      CreateAiInstance<Nnrt>(model_data, model_size, threads);
    }
#endif
    KAYLORDUT_LOG_INFO("number of instances is {}", this->instances_.size());
    for (size_t i = 0; i < instances_.size(); ++i) {
      tensors_data_.push_back(std::make_shared<ai_framework::TensorData>(
          instances_.at(i)->get_config()));
      auto it = instances_.at(i)->get_config().input_layer_shape.begin();
      auto model_height = it->second.at(2);
      yolo_preprocess_.push_back(
          std::make_shared<YoloPreProcess>(model_height, false));
      yolo_postprocess_.push_back(std::make_shared<YoloPostProcess>(
          instances_.at(i)->get_config(), conf_threshold, 0.6, 0.5));
      instances_.at(i)->BindInputAndOutput(*tensors_data_.at(i));
    }
  } catch (const std::bad_alloc &e) {
    KAYLORDUT_LOG_ERROR("Out of memory: {}", e.what());
    exit(EXIT_FAILURE);
  }
  instances_.at(0)->PrintLayerInfo();
  KAYLORDUT_LOG_INFO("Yolo thread pool initialization completed")
}

YoloThreadpool::YoloThreadpool(const char *model_data,
                               const uint64_t model_size,
                               const std::string model_extension,
                               std::vector<float> &conf_threshold,
                               int threads) {
  Initialize(model_data, model_size, model_extension, conf_threshold, threads);
}

YoloThreadpool::YoloThreadpool(std::string &model_path,
                               std::vector<float> &conf_threshold,
                               int threads) {
  auto model_extension = model_path.substr(model_path.find_last_of('.') + 1);
  KAYLORDUT_LOG_INFO("model_path is {},  model_extension is {}", model_path,
                     model_extension);
  std::ifstream file(model_path, std::ios::binary);
  assert(file.good());
  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *model_stream = new char[size];
  assert(model_stream);
  file.read(model_stream, size);
  file.close();
  Initialize(model_stream, size, model_extension, conf_threshold, threads);
  delete[] model_stream;
}

void YoloThreadpool::AddInferenceTask(
    const std::vector<cv::Mat> &original_image,
    const std::vector<double> timepoints, const bool clone_original_image,
    const std::vector<cv::Mat> &original_depth_image) {
  //  std::vector<cv::Mat> image;
  //  if (clone_original_image) {
  //    for (int i = 0; i < original_image.size(); ++i) {
  //      image.push_back(original_image.at(i).clone());
  //    }
  //  } else {
  //    image = original_image;
  //  }
  auto &image = original_image;
  auto &depth_image = original_depth_image;
  // lamda表达式传入参数需要使用值，不能使用引用，这里使用cv的智能指针，image可以获取保留数据的指针，如果使用引用的话，指向的数据会发生变化
  this->pool_->enqueue(
      [&](const std::vector<cv::Mat> image,
          const std::vector<cv::Mat> depth_image,
          const std::vector<double> time_points) {
        auto id = this->get_thread_id();
        std::lock_guard<std::mutex> lock_threads(this->threads_mutex_[id]);
        this->yolo_preprocess_.at(id)->Run(
            image, this->tensors_data_.at(id)->get_input_tensor_ptr());
        KAYLORDUT_TIME_COST_DEBUG("DoInference()",
                                  this->instances_.at(id)->DoInference());
        this->yolo_postprocess_.at(id)->Run(
            this->tensors_data_.at(id)->get_output_tensor_ptr());
        std::unique_lock<std::mutex> lock(this->result_queue_mutex_);
        if (!yolo_inference_result_queue_.empty()) {
          if (time_points.at(0) <=
              yolo_inference_result_queue_.back()->time_points.at(0)) {
            KAYLORDUT_LOG_WARN(
                "current time stamp is too old, drop the result");
            drop_count_++;
            return;
          }
        }
        auto res = std::make_shared<YoloInferenceResult>();
        res->time_points = time_points;
        auto now = get_current_time();
        res->time_points.push_back(now);
        res->original_image = image;
        res->results = this->yolo_postprocess_.at(id)->get_result();
        if (!depth_image.empty()) {
          res->original_depth_image_ = depth_image;
        }
        yolo_inference_result_queue_.push(res);
        result_ready_ = true;
        result_cv_.notify_one();
      },
      image, depth_image, timepoints);
}

std::shared_ptr<YoloThreadpool::YoloInferenceResult>
YoloThreadpool::GetInferenceResult() {
  std::lock_guard<std::mutex> lock(this->result_queue_mutex_);
  if (yolo_inference_result_queue_.empty()) {
    return nullptr;
  } else {
    auto res = this->yolo_inference_result_queue_.front();
    this->yolo_inference_result_queue_.pop();
    return res;
  }
}

int YoloThreadpool::get_result_queue_size() {
  std::lock_guard<std::mutex> lock(this->result_queue_mutex_);
  return this->yolo_inference_result_queue_.size();
}

int YoloThreadpool::get_thread_id() {
  std::lock_guard<std::mutex> lock(id_mutex_);
  auto id = id_;
  ++id_;
  if (id_ == num_threads_) {
    id_ = 0;
  }
  return id;
}