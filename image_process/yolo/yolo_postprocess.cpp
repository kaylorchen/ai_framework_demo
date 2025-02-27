//
// Created by kaylor on 6/14/24.
//

#include "yolo_postprocess.h"

#include <cmath>
#include <set>

#include "iostream"
#include "kaylordut/log/logger.h"

static bool ContainsSubString(const std::string &str,
                              const std::string &substring) {
  return str.find(substring) != std::string::npos;
}

inline static int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

YoloPostProcess::YoloPostProcess(const ai_framework::Config &config,
                                 std::vector<float> &conf_threshold,
                                 float sum_conf_threshold,
                                 float iou_threshold) {
  model_format_ = config.model_format;
  iou_threshold_ = iou_threshold;
  num_of_layers_ = config.output_tensors_count;
  for (auto &kv : config.output_element_count) {
    output_layer_names_.push_back(kv.first);
    output_layer_shape.push_back(config.output_layer_shape.at(kv.first));
    output_element_count_.push_back(kv.second);
    if (config.zero_point.find(kv.first) != config.zero_point.end()) {
      zero_points_.push_back(config.zero_point.at(kv.first));
    }
    if (config.scale.find(kv.first) != config.scale.end()) {
      scale_.push_back(config.scale.at(kv.first));
    }
  }
  conf_threshold_ = &conf_threshold;
  sum_conf_threshold_ = sum_conf_threshold;
  auto input_shape =
      config.input_layer_shape.at(config.input_index_to_name.at(0));
  model_width_ = input_shape.at(3);
  model_height_ = input_shape.at(2);
  auto output_boxes_shape =
      config.output_layer_shape.at(config.output_index_to_name.at(0));
  dfl_len_ = output_boxes_shape.at(1) / 4;
  auto output_boxes_name = config.output_index_to_name.at(0);
  if (ContainsSubString(output_boxes_name, "yolov10")) {
    model_type_ = ModelType::DETECTION_V10;
  } else if (ContainsSubString(output_boxes_name, "yolov8_detect")) {
    model_type_ = ModelType::DETECTION_V8;
  } else if (ContainsSubString(output_boxes_name, "yolov8_pose")) {
    model_type_ = ModelType::POSE_V8;
  } else if (ContainsSubString(output_boxes_name, "yolo11_detect")) {
    model_type_ = ModelType::DETECTION_V11;
  } else if (ContainsSubString(output_boxes_name, "yolo11_segment")) {
    model_type_ = ModelType::SEGMENT_V11;
    auto tmp = config.output_layer_shape.rbegin();
    seg_width_ = tmp->second.at(3);
    seg_height_ = tmp->second.at(2);
    KAYLORDUT_LOG_INFO("yolo11_segment resolution: {}x{}", seg_width_,
                       seg_height_);
  }
}

void YoloPostProcess::Run(void **&tensors) {
  result_.clear();
  bboxes_.clear();
  class_id_.clear();
  obj_probs_.clear();
  bboxes_idx_.clear();
  if (model_type_ == ModelType::DETECTION_V8 ||
      model_type_ == ModelType::DETECTION_V10 ||
      model_type_ == ModelType::DETECTION_V11 ||
      model_type_ == ModelType::SEGMENT_V11) {
    PostProcessDetectSegment(tensors);
  } else if (model_type_ == ModelType::POSE_V8) {
    kpt.clear();
    visibilities.clear();
    PostProcessPose(tensors);
  }
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left,
                                     int right, std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                              float ymax0, float xmin1, float ymin1,
                              float xmax1, float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<YoloPostProcess::Bbox> &bboxes,
               std::vector<int> &order, float threshold) {
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1) {
        continue;
      }
      float xmin0 = bboxes.at(n).x1;
      float ymin0 = bboxes.at(n).y1;
      float xmax0 = bboxes.at(n).x2;
      float ymax0 = bboxes.at(n).y2;
      float xmin1 = bboxes.at(m).x1;
      float ymin1 = bboxes.at(m).y1;
      float xmax1 = bboxes.at(m).x2;
      float ymax1 = bboxes.at(m).y2;
      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);
      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int nms(const int validCount,
               const std::vector<YoloPostProcess::Bbox> &bboxes,
               const std::vector<int> classIds, std::vector<int> &order,
               const int filterId, const float threshold) {
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[order[i]] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[order[j]] != filterId) {
        continue;
      }
      float xmin0 = bboxes.at(n).x1;
      float ymin0 = bboxes.at(n).y1;
      float xmax0 = bboxes.at(n).x2;
      float ymax0 = bboxes.at(n).y2;
      float xmin1 = bboxes.at(m).x1;
      float ymin1 = bboxes.at(m).y1;
      float xmax1 = bboxes.at(m).x2;
      float ymax1 = bboxes.at(m).y2;
      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);
      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

void YoloPostProcess::PostProcessPose(void **&tensors) {
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int output_per_branch = num_of_layers_ / 3;
  for (int i = 0; i < 3; ++i) {
    int box_idx = i * output_per_branch;
    int score_idx = i * output_per_branch + 1;
    int kpt_idx = i * output_per_branch + 2;
    int visibilities_idx = i * output_per_branch + 3;
    grid_h = output_layer_shape.at(box_idx).at(2);
    grid_w = output_layer_shape.at(box_idx).at(3);
    stride = model_height_ / grid_h;
    validCount +=
        ProcessPose(reinterpret_cast<const float *>(tensors[box_idx]),
                    reinterpret_cast<const float *>(tensors[score_idx]),
                    reinterpret_cast<const float *>(tensors[kpt_idx]),
                    reinterpret_cast<const float *>(tensors[visibilities_idx]),
                    grid_w, grid_h, stride);
  }
  if (validCount <= 0) {
    return;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; i++) {
    indexArray.push_back(i);
  }
  quick_sort_indice_inverse(obj_probs_, 0, validCount - 1, indexArray);
  nms(validCount, bboxes_, indexArray, iou_threshold_);
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1) {
      continue;
    }
    int n = indexArray[i];
    Result res;
    res.model_type = ModelType::POSE_V8;
    res.box = bboxes_.at(n);
    res.class_id = class_id_.at(n);
    res.obj_prob = obj_probs_.at(n);
    for (int j = 0; j < 34; j += 2) {
      auto kpt_x = kpt.at(34 * n + j);
      auto kpt_y = kpt.at(34 * n + j + 1);
      auto visibility = visibilities.at(17 * n + j / 2);
      res.key_points[j / 2].x = kpt_x;
      res.key_points[j / 2].y = kpt_y;
      res.key_points[j / 2].visibility = visibility;
    }
    result_.push_back(res);
  }
}

void YoloPostProcess::PostProcessDetectSegment(void **&tensors) {
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int output_per_branch = num_of_layers_ / 3;
  for (int i = 0; i < 3; ++i) {
    int box_index = i * output_per_branch;
    int score_index = i * output_per_branch + 1;
    int sum_score_index = i * output_per_branch + 2;
    grid_h = output_layer_shape.at(box_index).at(2);
    grid_w = output_layer_shape.at(box_index).at(3);
    stride = model_height_ / grid_h;
    validCount += ProcessDetect(tensors[box_index], tensors[score_index],
                                tensors[sum_score_index], grid_w, grid_h,
                                stride, i, output_per_branch);
  }
  //    std::cout << "validCount: " << validCount << std::endl;
  //  KAYLORDUT_LOG_DEBUG("validCount: {}", validCount);
  if (validCount <= 0) {
    return;
  }
  std::vector<int> indexArray;
  if (model_type_ == ModelType::DETECTION_V8 ||
      model_type_ == ModelType::DETECTION_V11 ||
      model_type_ == ModelType::SEGMENT_V11) {
    for (int i = 0; i < validCount; i++) {
      indexArray.push_back(i);
    }
    quick_sort_indice_inverse(obj_probs_, 0, validCount - 1, indexArray);
    std::set<int> class_set(std::begin(class_id_), std::end(class_id_));
    for (auto c : class_set) {
      nms(validCount, bboxes_, class_id_, indexArray, c, iou_threshold_);
    }
  }
  for (int i = 0; i < validCount; ++i) {
    int n = i;
    Result res;
    if (model_type_ == ModelType::DETECTION_V8 ||
        model_type_ == ModelType::DETECTION_V11 ||
        model_type_ == ModelType::SEGMENT_V11) {
      if (indexArray[i] == -1) {
        continue;
      }
      n = indexArray[i];
      res.model_type = model_type_;
    } else if (model_type_ == ModelType::DETECTION_V10) {
      n = i;
      res.model_type = model_type_;
    }
    res.box = bboxes_.at(n);
    res.class_id = class_id_.at(n);
    // 上面快排的时候，元素有交换
    res.obj_prob = obj_probs_.at(i);
    if (model_type_ == ModelType::SEGMENT_V11) {
      int mask_index = bboxes_idx_.at(n).index * output_per_branch + 3;
      int proto_index = num_of_layers_ - 1;
      ProcessSegment(tensors[mask_index], tensors[proto_index], res,
                     bboxes_idx_.at(n), output_per_branch);
    }
    result_.push_back(res);
  }
}

void YoloPostProcess::ProcessSegment(const void *mask_tensor,
                                     const void *proto_tensor, Result &result,
                                     BboxesIdx bboxes_idx,
                                     int output_per_branch) {
  const float *mask_tensor_float = reinterpret_cast<const float *>(mask_tensor);
  const float *proto_tensor_float =
      reinterpret_cast<const float *>(proto_tensor);
  const int8_t *mask_tensor_int8 =
      reinterpret_cast<const int8_t *>(mask_tensor);
  const int8_t *proto_tensor_int8 =
      reinterpret_cast<const int8_t *>(proto_tensor);
  bool is_qnt = zero_points_.empty() ? false : true;
#define COUNT 32
  float mask[COUNT] = {0};
  cv::Mat proto;
  if (model_format_ != ModelFormat::RKNN_FORMAT) {
    for (int i = 0; i < COUNT; i++) {
      mask[i] =
          mask_tensor_float[bboxes_idx.sub_index + i * bboxes_idx.grid_len];
    }
    proto = cv::Mat(COUNT, seg_height_ * seg_width_, CV_32FC1,
                    const_cast<float *>(proto_tensor_float));
  } else {
    auto mask_zp = zero_points_.at(bboxes_idx.index * output_per_branch + 3);
    auto mask_scale = scale_.at(bboxes_idx.index * output_per_branch + 3);
    auto proto_zp = zero_points_.at(num_of_layers_ - 1);
    auto proto_scale = scale_.at(num_of_layers_ - 1);
    // KAYLORDUT_LOG_INFO("mask: {}, {}, proto: {} {}", mask_zp, mask_scale,
    //                    proto_zp, proto_scale);
    for (int i = 0; i < COUNT; ++i) {
      auto &element =
          mask_tensor_int8[bboxes_idx.sub_index + i * bboxes_idx.grid_len];
      mask[i] =
          is_qnt ? deqnt_affine_to_f32(element, mask_zp, mask_scale) : 0.0f;
    }
    proto = cv::Mat::zeros(COUNT, seg_height_ * seg_width_, CV_32FC1);
    if (is_qnt) {
      float *proto_ptr = proto.ptr<float>();
      for (int i = 0; i < COUNT * seg_height_ * seg_width_; ++i) {
        auto &element = proto_tensor_int8[i];
        proto_ptr[i] = deqnt_affine_to_f32(element, proto_zp, proto_scale);
      }
    }
  }
  cv::Mat mask_mat = cv::Mat(1, COUNT, CV_32FC1, mask);
  cv::Mat res_mat = mask_mat * proto;
  res_mat = res_mat.reshape(0, seg_height_);
  auto image_scale = model_height_ / seg_height_;
  auto roi = cv::Rect(result.box.x1 / image_scale, result.box.y1 / image_scale,
                      (result.box.x2 - result.box.x1) / image_scale,
                      (result.box.y2 - result.box.y1) / image_scale);
  auto sub_image = res_mat(roi);
  cv::Mat sub_res_image = sub_image > 0.5;
  result.seg_mat = cv::Mat::zeros(res_mat.size(), CV_8UC1);
  sub_res_image.copyTo(result.seg_mat(roi));
}

static void compute_dfl(float *tensor, int dfl_len, float *box) {
  for (int b = 0; b < 4; b++) {
    float exp_t[dfl_len];
    float exp_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < dfl_len; i++) {
      exp_t[i] = exp(tensor[i + b * dfl_len]);
      exp_sum += exp_t[i];
    }

    for (int i = 0; i < dfl_len; i++) {
      acc_sum += exp_t[i] / exp_sum * i;
    }
    box[b] = acc_sum;
  }
}

uint16_t YoloPostProcess::ProcessPose(const float *box_tensor,
                                      const float *score_tensor,
                                      const float *kpt_tensor,
                                      const float *visibility_tensor,
                                      int grid_w, int grid_h, int stride) {
  uint16_t valid_count = 0;
  int grid_len = grid_w * grid_h;
  for (int i = 0; i < grid_h; ++i) {
    for (int j = 0; j < grid_w; ++j) {
      int offset = i * grid_w + j;
      float max_score = 0;
      int max_class_id = -1;
      for (size_t k = 0; k < conf_threshold_->size(); ++k) {
        if (score_tensor[offset] > conf_threshold_->at(k) &&
            score_tensor[offset] > max_score) {
          max_score = score_tensor[offset];
          max_class_id = k;
        }
        offset += grid_len;
      }
      if (max_class_id != -1) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len_ * 4];
        for (int k = 0; k < dfl_len_ * 4; ++k) {
          before_dfl[k] = box_tensor[offset];
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len_, box);
        Bbox _bbox;
        _bbox.x1 = (-box[0] + j + 0.5) * stride;
        _bbox.y1 = (-box[1] + i + 0.5) * stride;
        _bbox.x2 = (box[2] + j + 0.5) * stride;
        _bbox.y2 = (box[3] + i + 0.5) * stride;
        bboxes_.push_back(_bbox);
        obj_probs_.push_back(max_score);
        class_id_.push_back(max_class_id);
        offset = i * grid_w + j;
        for (int k = 0; k < 17; ++k) {
          auto kpt_x = *(kpt_tensor + offset + 2 * k * grid_len);
          auto kpt_y = *(kpt_tensor + offset + (2 * k + 1) * grid_len);
          auto kpt_visibility = *(visibility_tensor + offset + k * grid_len);
          kpt.push_back(kpt_x);
          kpt.push_back(kpt_y);
          visibilities.push_back(kpt_visibility);
        }
        valid_count++;
      }
    }
  }
  return valid_count;
}

uint16_t YoloPostProcess::ProcessDetect(const void *box_tensor,
                                        const void *score_tensor,
                                        const void *sum_score_tensor,
                                        int grid_w, int grid_h, int stride,
                                        int index, int output_per_branch) {
  const float *box_tensor_float = reinterpret_cast<const float *>(box_tensor);
  const float *score_tensor_float =
      reinterpret_cast<const float *>(score_tensor);
  const float *sum_score_tensor_float =
      reinterpret_cast<const float *>(sum_score_tensor);
  const int8_t *box_tensor_int8 = reinterpret_cast<const int8_t *>(box_tensor);
  const int8_t *score_tensor_int8 =
      reinterpret_cast<const int8_t *>(score_tensor);
  const int8_t *sum_score_tensor_int8 =
      reinterpret_cast<const int8_t *>(sum_score_tensor);
  bool is_qnt = zero_points_.empty() ? false : true;
  int8_t score_sum_thres_i8 =
      is_qnt ? qnt_f32_to_affine(sum_conf_threshold_,
                                 zero_points_.at(index * output_per_branch + 2),
                                 scale_.at(index * output_per_branch + 2))
             : 0;

  uint16_t valid_count = 0;
  int grid_len = grid_w * grid_h;
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; ++j) {
      int offset = i * grid_w + j;
      if (model_format_ == ModelFormat::ONNX_FORMAT ||
          model_format_ == ModelFormat::TRT_FORMAT ||
          model_format_ == ModelFormat::NNRT_FORMAT) {
        if (sum_score_tensor_float[offset] < sum_conf_threshold_) {
          continue;
        }
      } else if (model_format_ == ModelFormat::RKNN_FORMAT) {
        if (sum_score_tensor_int8[offset] < score_sum_thres_i8) {
          continue;
        }
      }
      float max_score_float = 0;
      int8_t max_score_int8 =
          is_qnt ? qnt_f32_to_affine(
                       0.0f, zero_points_.at(index * output_per_branch + 1),
                       scale_.at(index * output_per_branch + 1))
                 : 0;
      int max_class_id = -1;
      for (size_t k = 0; k < conf_threshold_->size(); ++k) {
        if (model_format_ == ModelFormat::ONNX_FORMAT ||
            model_format_ == ModelFormat::TRT_FORMAT ||
            model_format_ == ModelFormat::NNRT_FORMAT) {
          if (score_tensor_float[offset] > conf_threshold_->at(k) &&
              score_tensor_float[offset] > max_score_float) {
            max_score_float = score_tensor_float[offset];
            max_class_id = k;
          }
        } else if (model_format_ == ModelFormat::RKNN_FORMAT) {
          auto score_thres_i8 =
              is_qnt ? qnt_f32_to_affine(
                           conf_threshold_->at(k),
                           zero_points_.at(index * output_per_branch + 1),
                           scale_.at(index * output_per_branch + 1))
                     : 0;
          if (score_tensor_int8[offset] > score_thres_i8 &&
              score_tensor_int8[offset] > max_score_float) {
            max_score_int8 = score_tensor_int8[offset];
            max_class_id = k;
          }
        }
        offset += grid_len;
      }
      if (max_class_id != -1) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len_ * 4];
        for (int k = 0; k < dfl_len_ * 4; ++k) {
          if (model_format_ == ModelFormat::ONNX_FORMAT ||
              model_format_ == ModelFormat::TRT_FORMAT ||
              model_format_ == ModelFormat::NNRT_FORMAT) {
            before_dfl[k] = box_tensor_float[offset];
          } else if (model_format_ == ModelFormat::RKNN_FORMAT) {
            before_dfl[k] =
                is_qnt ? deqnt_affine_to_f32(
                             box_tensor_int8[offset],
                             zero_points_.at(index * output_per_branch),
                             scale_.at(index * output_per_branch))
                       : 0;
          }
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len_, box);
        Bbox _bbox;
        _bbox.x1 = (-box[0] + j + 0.5) * stride;
        _bbox.y1 = (-box[1] + i + 0.5) * stride;
        _bbox.x2 = (box[2] + j + 0.5) * stride;
        _bbox.y2 = (box[3] + i + 0.5) * stride;
        int width_pixel_delta = 1;
        int height_pixel_delta = 1;
        if (model_type_ == ModelType::SEGMENT_V11) {
          height_pixel_delta = model_height_ / seg_height_;
          width_pixel_delta = model_width_ / seg_width_;
        }
        if (std::abs(_bbox.x1 - _bbox.x2) < width_pixel_delta) {
          KAYLORDUT_LOG_WARN("bbox width is too small: {}",
                             std::abs(_bbox.x1 - _bbox.x2));
          continue;
        }
        if (std::abs(_bbox.y1 - _bbox.y2) < height_pixel_delta) {
          KAYLORDUT_LOG_WARN("bbox height is too small: {}",
                             std::abs(_bbox.y1 - _bbox.y2));
          continue;
        }
        bboxes_.push_back(_bbox);
        if (model_format_ == ModelFormat::ONNX_FORMAT ||
            model_format_ == ModelFormat::TRT_FORMAT ||
            model_format_ == ModelFormat::NNRT_FORMAT) {
          obj_probs_.push_back(max_score_float);
        } else if (model_format_ == ModelFormat::RKNN_FORMAT) {
          auto max_score =
              is_qnt ? deqnt_affine_to_f32(
                           max_score_int8,
                           zero_points_.at(index * output_per_branch + 1),
                           scale_.at(index * output_per_branch + 1))
                     : 0;
          obj_probs_.push_back(max_score);
          //                              KAYLORDUT_LOG_INFO("max_score: {} {},
          //                              id: {}, box[{} {} {} {}]",
          //                                                 max_score,
          //                                                 max_score_int8,
          //                                                 max_class_id,
          //                                                 _bbox.x1, _bbox.y1,
          //                                                 _bbox.x2,
          //                                                 _bbox.y2);
        }
        class_id_.push_back(max_class_id);
        bboxes_idx_.push_back({index, i * grid_w + j, grid_len});
        valid_count++;
      }
    }
  }
  return valid_count;
}
