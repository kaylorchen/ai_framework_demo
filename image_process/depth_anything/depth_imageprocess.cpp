//
// Created by kaylor on 7/10/24.
//

#include "depth_imageprocess.h"

DepthImageprocess::DepthImageprocess(const ai_framework::Config &config,
                                     int target_size,
                                     std::vector<float> mean_values,
                                     std::vector<float> std_values,
                                     bool debug) {
  model_format_ = config.model_format;
  for (const auto &kv : config.input_layer_shape) {
    input_shape_.push_back(kv.second);
#ifdef RK3588
    input_width_equal_stride_.push_back(config.width_equal_stride.at(kv.first));
    input_stride_.push_back(config.stride.at(kv.first));
#endif
  }
  for (const auto &kv : config.output_layer_shape) {
    output_zero_points_.push_back(config.zero_point.at(kv.first));
    output_scale_.push_back(config.scale.at(kv.first));
  }
  target_size_ = target_size;
  mean_values_ = mean_values;
  std_values_ = std_values;
  debug_ = debug;
}

void DepthImageprocess::PreProcess(const std::vector<cv::Mat> &input,
                                   void **tenssors) {
  for (size_t i = 0; i < input.size(); ++i) {
    auto &original_input = input.at(i);
    int resize_width = target_size_;
    int resize_height = target_size_;
    if (original_input.cols >= original_input.rows) {
      float scale = 1.0f * original_input.cols / target_size_;
      resize_height = original_input.rows / scale;
    } else {
      float scale = 1.0f * original_input.rows / target_size_;
      resize_width = original_input.cols / scale;
    }
    cv::Mat dst;
    cv::resize(original_input, dst, cv::Size(resize_width, resize_height),
               cv::INTER_NEAREST);
    resize_mat_size_.push_back(cv::Size(resize_width, resize_height));
    input_mat_size_.push_back(original_input.size());
#if RK3588
    cv::Mat cvt;
    cv::cvtColor(dst, cvt, cv::COLOR_BGR2RGB);
    cv::Mat res;
    if (input_width_equal_stride_.at(i)) {
      res =
          cv::Mat(target_size_, target_size_, dst.type(), (void *)tenssors[i]);
    }
    MakeSquare(cvt, res);
    if (!input_width_equal_stride_.at(i)) {
      auto width = input_shape_.at(i).at(2);
      auto height = input_shape_.at(i).at(1);
      auto channel = input_shape_.at(i).at(3);
      // copy from src to dst with stride
      uint8_t *src_ptr = res.ptr();
      uint8_t *dst_ptr = (uint8_t *)tenssors[i];
      // width-channel elements
      auto src_wc_elems = width * channel * sizeof(uint8_t);
      auto dst_wc_elems = input_stride_.at(i) * channel * sizeof(uint8_t);
      for (int b = 0; b < input_shape_.at(i).at(0); b++) {
        for (int h = 0; h < height; ++h) {
          memcpy(dst_ptr, src_ptr, src_wc_elems);
          src_ptr += src_wc_elems;
          dst_ptr += dst_wc_elems;
        }
      }
    }
#else
    cv::Mat res;
    MakeSquare(dst, res);
    PopulateData(res, reinterpret_cast<float *>(tenssors[i]));
#endif
    if (debug_) {
      cv::imshow("PreProcess Image", dst);
      cv::waitKey(1);
    }
  }
}

void DepthImageprocess::MakeSquare(const cv::Mat &src, cv::Mat &dst) {
  // 获取图像的宽和高
  int width = src.cols;
  int height = src.rows;
  // 计算需要填充的尺寸
  int border_left = 0;
  int border_right = 0;
  int border_top = 0;
  int border_bottom = 0;

  if (height > width) {
    int delta = height - width;
    border_left += (delta >> 1);
    border_right = border_left;
  } else {
    int delta = width - height;
    border_top += (delta >> 1);
    border_bottom = border_top;
  }
  // 使用灰色(0,0,0)填充边缘
  cv::copyMakeBorder(src, dst, border_top, border_bottom, border_left,
                     border_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

uint64_t DepthImageprocess::PopulateData(const cv::Mat &data, float *dst) {
  if (data.channels() != 3 || data.type() != CV_8UC3) {
    return 0;
  }
  auto *R = dst;
  auto *G = dst + data.total();
  auto *B = dst + data.total() * 2;
  for (int i = 0; i < data.rows; ++i) {
    for (int j = 0; j < data.cols; ++j) {
      // Mat 的数据是BGR
      *B = (data.at<cv::Vec3b>(i, j)[0] / 255.0f - mean_values_.at(2)) /
           std_values_.at(2);
      B++;
      *G = (data.at<cv::Vec3b>(i, j)[1] / 255.0f - mean_values_.at(1)) /
           std_values_.at(1);
      G++;
      *R = (data.at<cv::Vec3b>(i, j)[2] / 255.0f - mean_values_.at(0)) /
           std_values_.at(0);
      R++;
    }
  }
  // 返回填充的字节数
  return data.total() * data.channels() * sizeof(*dst);
}

bool DepthImageprocess::GetResult(cv::Mat &result) {
  bool ret = false;
  std::lock_guard<std::mutex> lock(depth_queue_mutex_);
  if (!depth_queue_.empty()) {
    ret = true;
    result = depth_queue_.front();
    depth_queue_.pop();
  }
  return ret;
}

cv::Mat cropImageCenter(const cv::Mat &image, const cv::Size &targetSize) {
  int width = targetSize.width;
  int height = targetSize.height;
  // 如果目标尺寸大于原始图像尺寸，直接返回原始图像
  if (width > image.cols || height > image.rows) {
    std::cerr << "Target size is larger than the original image size."
              << std::endl;
    return image;
  }
  // 计算左上角坐标
  int x = (image.cols - width) / 2;
  int y = (image.rows - height) / 2;
  // 定义一个感兴趣区域 (ROI)
  cv::Rect roi(x, y, width, height);
  // 截取图像
  cv::Mat croppedImage = image(roi).clone();
  return croppedImage;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

void DepthImageprocess::PostProcess(void **tensors) {
  cv::Mat depth;
  if (model_format_ == ModelFormat::RKNN_FORMAT) {
    depth = cv::Mat(target_size_, target_size_, CV_32FC1);
    auto data_ptr = (float *)depth.ptr();
    auto tensors_int8 = reinterpret_cast<int8_t **>(tensors);
    for (int i = 0; i < target_size_ * target_size_; ++i) {
      *data_ptr = deqnt_affine_to_f32(
          tensors_int8[0][i], output_zero_points_.at(0), output_scale_.at(0));
      data_ptr++;
    }
  } else if (model_format_ == ModelFormat::TRT_FORMAT ||
             model_format_ == ModelFormat::ONNX_FORMAT ||
             model_format_ == ModelFormat::NNRT_FORMAT) {
    depth = cv::Mat(target_size_, target_size_, CV_32FC1, tensors[0]);
  }
  cv::Mat resized_depth = cropImageCenter(depth, resize_mat_size_.at(0));
  cv::Mat depth_normalized;
  cv::normalize(resized_depth, depth_normalized, 0, 255, cv::NORM_MINMAX,
                CV_8UC1);
  cv::Mat depth_colored;
  cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);
  cv::Mat result;
  cv::resize(depth_colored, result, input_mat_size_.at(0));
  std::lock_guard<std::mutex> lock_guard(depth_queue_mutex_);
  depth_queue_.emplace(result);
}
