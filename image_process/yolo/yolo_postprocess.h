//
// Created by kaylor on 6/14/24.
//

#pragma once
#include "ai_instance.h"
#include "opencv2/opencv.hpp"
#include "stdint.h"

class YoloPostProcess {
 public:
   enum ModelType : uint16_t {
     DETECTION_V8 = 0,
     DETECTION_V10,
     DETECTION_V11,
     DETECTION_V13,
     POSE_V8,
     SEGMENT_V11,
     DETECTION_V26,
   };
   struct Bbox {
     float x1;
     float y1;
     float x2;
     float y2;
  };
  struct KeyPoint {
    float x;
    float y;
    float visibility;
  };
  struct Result {
    ModelType model_type;
    Bbox box;
    float obj_prob;
    int class_id;
    cv::Mat seg_mat;
    KeyPoint key_points[17];
  };
  typedef struct {
    int index;
    int sub_index;
    int grid_len;
  } BboxesIdx;
  YoloPostProcess() = delete;
  YoloPostProcess(const ai_framework::Config &config,
                  std::vector<float> &conf_threshold, float sum_conf_threshold,
                  float iou_threshold = 0.5f);
  void Run(void **&tensors);
  const std::vector<Result> &get_result() const { return result_; }

 private:
  void PostProcessDetectSegment(void **&tensors);
  void ProcessSegment(const void *mask_tensor, const void *proto_tensor,
                      Result &result, BboxesIdx bboxes_idx,
                      int output_per_branch);
  void PostProcessPose(void **&tensors);
  uint16_t ProcessDetect(const void *box_tensor, const void *score_tensor,
                         const void *sum_score_tensor, int grid_w, int grid_h,
                         int stride, int index, int output_per_branch);
  uint16_t ProcessPose(const void *box_tensor, const void *score_tensor,
                       const void *kpt_tensor, const void *visibility_tensor,
                       int grid_w, int grid_h, int stride, int index,
                       int output_per_branch);
  uint16_t num_of_layers_;
  std::vector<std::string> output_layer_names_;
  std::vector<size_t> output_element_count_;
  std::vector<std::vector<int64_t>> output_layer_shape;
  std::vector<float> zero_points_;
  std::vector<float> scale_;
  std::vector<float> *conf_threshold_;
  float sum_conf_threshold_;
  int model_width_{0};
  int model_height_{0};
  int seg_width_{0};
  int seg_height_{0};
  int dfl_len_{0};
  std::vector<Bbox> bboxes_;
  std::vector<BboxesIdx> bboxes_idx_;
  std::vector<int> class_id_;
  std::vector<float> obj_probs_;
  std::vector<float> kpt;
  std::vector<float> visibilities;
  std::vector<Result> result_;
  ModelType model_type_;
  ModelFormat model_format_;
  float iou_threshold_;
};
