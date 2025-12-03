//
// Created by kaylor on 11/25/25.
//

#include "stereo_image_process.h"

#include <kaylordut/log/logger.h>

StereoImageProcess::StereoImageProcess(
    const ai_framework::Config &config, const std::vector<float> &K,
    float baseline, float doffs) {
  std::string left = config.input_index_to_name.at(0);
  width_ = config.input_layer_shape.at(left).at(3);
  height_ = config.input_layer_shape.at(left).at(2);
  K_ = K;
  baseline_ = baseline;
  doffs_ = doffs;
  auto size = config.input_single_element_size.at(left);
  if (size == sizeof(float)) {
    input_data_type_ = CV_32FC3;
    padder_ = std::make_shared<image_processing::ImagePadder>(
        32, image_processing::PaddingMode::REPLICATE);
  } else if (size == sizeof(uint8_t)) {
    input_data_type_ = CV_8UC3;
    padder_ = std::make_shared<image_processing::ImagePadder>(
        32, image_processing::PaddingMode::ADAPTIVE);
  }
}

void StereoImageProcess::FillData(void **&tensors,
                                            long unsigned int i,
                                            cv::Mat img_resized,
                                            int data_type) {
  // 转换为指定类型
  cv::Mat img;
  if (img_resized.type() == data_type) {
    img = img_resized;
  } else {
    img_resized.convertTo(img, data_type, 1.0f);
  }

  // 分离通道
  std::vector<cv::Mat> bgr_channels;
  cv::split(img, bgr_channels);

  // 使用 uint8_t* 作为通用字节指针（支持指针算术）
  uint8_t *dst = static_cast<uint8_t *>(tensors[i]);

  int size_per_pixel;
  if (data_type == CV_32FC3) {
    size_per_pixel = sizeof(float);
  } else if (data_type == CV_8UC3) {
    size_per_pixel = sizeof(uint8_t);
  } else {
    // 可选：抛出异常或返回错误
    throw std::invalid_argument("Unsupported data_type");
  }

  const int total_pixels = img.total();
  const int channel_size = total_pixels * size_per_pixel;

  // 按 RGB 顺序写入：R, G, B
  memcpy(dst, bgr_channels[2].data, channel_size);                    // R
  memcpy(dst + channel_size, bgr_channels[1].data, channel_size);     // G
  memcpy(dst + channel_size * 2, bgr_channels[0].data, channel_size); // B
}

std::shared_ptr<StereoImageProcess::PreProcessResult>
StereoImageProcess::PreProcess(const std::vector<cv::Mat> &imgs,
                                         void **&tensors) {
  auto result = std::make_shared<PreProcessResult>();
  for (long unsigned int i = 0; i < imgs.size(); ++i) {
    auto aspect_ratio_img =
        ResizeKeepAspectRatio(imgs.at(i), cv::Size(width_, height_));
    if (i == 0) {
      *result = aspect_ratio_img;
      KAYLORDUT_LOG_INFO_ONCE("resized scale = {}", aspect_ratio_img.scale);
    }
    KAYLORDUT_LOG_INFO_ONCE(
        "original_img_size = {}x{} --> aspect_ratio_size = {}x{}",
        aspect_ratio_img.original_img.cols, aspect_ratio_img.original_img.rows,
        aspect_ratio_img.resized_img.cols, aspect_ratio_img.resized_img.rows);
    cv::Mat img_padded =
        padder_->Pad(aspect_ratio_img.resized_img, true, cv::Size(width_, height_));
    KAYLORDUT_LOG_INFO_ONCE("aspect_ratio_img_size = {}x{} --> padded_size = {}x{}",
                            aspect_ratio_img.resized_img.cols,
                            aspect_ratio_img.resized_img.rows, img_padded.cols,
                            img_padded.rows);
    // cv::imwrite("pad.png", img_padded);
    // exit(0);
    cv::Mat img_resized;
    if (img_padded.size() != cv::Size(width_, height_)) {
      cv::resize(img_padded, img_resized, cv::Size(width_, height_));
    } else {
      img_resized = img_padded;
    }
    KAYLORDUT_LOG_INFO_ONCE("img_padded_size = {}x{} --> target_size = {}x{}",
                            img_padded.cols, img_padded.rows, width_, height_);

    FillData(tensors, i, img_resized, input_data_type_);
  }
  return result;
}

std::shared_ptr<StereoImageProcess::PostProcessResult>
StereoImageProcess::PostProcess(
    void **&tensors, const PreProcessResult &pre_process_result) {
  auto result = std::make_shared<PostProcessResult>();
  result->original_img = pre_process_result.original_img;
  auto inference_output = cv::Mat(height_, width_, CV_32FC1, tensors[0]);
  KAYLORDUT_LOG_INFO_ONCE("inference_output_size = {}x{}",
                          inference_output.cols, inference_output.rows);
  result->disparity_img = padder_->Unpad(inference_output);
  KAYLORDUT_LOG_INFO_ONCE("disparity_img_size = {}x{}",
                          result->disparity_img.cols,
                          result->disparity_img.rows);
  RemoveInvisiblePoints(result->disparity_img);
  result->depth_img = cv::Mat(result->disparity_img.size(), CV_32FC1);
  std::vector<float> K = K_;
  K[0] *= pre_process_result.scale;
  K[4] *= pre_process_result.scale;
  K[2] *= pre_process_result.scale;
  K[5] *= pre_process_result.scale;
  float doffs = pre_process_result.scale * doffs_;
  ComputeDepth(result->depth_img, result->disparity_img, K[0], baseline_,
               doffs);
  result->cloud = DepthImageToPointCloud(result->depth_img,
                                         pre_process_result.resized_img, K);
  return result;
}

void StereoImageProcess::RemoveInvisiblePoints(cv::Mat &disp) {
  if (disp.empty())
    return;
  cv::parallel_for_(cv::Range(0, disp.rows), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; y++) {
      float *ptr = disp.ptr<float>(y);
      for (int x = 0; x < disp.cols; x++) {
        float us_right = static_cast<float>(x) - ptr[x];
        if (us_right < 0) {
          ptr[x] = std::numeric_limits<float>::infinity();
        }
      }
    }
  });
}

void StereoImageProcess::ComputeDepth(cv::Mat &depth,
                                                const cv::Mat &disp, float K00,
                                                float baseline, float doffs) {
  if (disp.empty())
    return;
  // 预计算常量
  float const_number = K00 * baseline;
  cv::parallel_for_(cv::Range(0, disp.rows), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; y++) {
      const float *disp_ptr = disp.ptr<float>(y);
      float *depth_ptr = depth.ptr<float>(y);

      for (int x = 0; x < disp.cols; x++) {
        float d = disp_ptr[x];
        // 避免除以零和无效视差
        if (d > 0 && std::isfinite(d)) {
          depth_ptr[x] = const_number / (d + doffs);
        } else {
          depth_ptr[x] = 0.0f; // 或者设为其他无效值
        }
      }
    }
  });
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
StereoImageProcess::DepthImageToPointCloud(const cv::Mat &depth_img,
                                                     const cv::Mat &rgb_img,
                                                     std::vector<float> &K) {
  if (K.size() != 9) {
    KAYLORDUT_LOG_ERROR("K.size() != 9");
    return nullptr;
  }
  auto fx = K[0];
  auto fy = K[4];
  auto cx = K[2];
  auto cy = K[5];
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  float depth = 0.0f;
  cloud->points.reserve(depth_img.rows * depth_img.cols);
  for (int y = 0; y < depth_img.rows; ++y) {
    for (int x = 0; x < depth_img.cols; ++x) {
      depth = depth_img.at<float>(y, x);
      if (depth > 0.0f) {
        pcl::PointXYZRGB point;
        point.x = (x - cx) * depth / fx;
        point.y = (y - cy) * depth / fy;
        point.z = depth;
        auto color = rgb_img.at<cv::Vec3b>(y, x);
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
        cloud->push_back(point);
      }
    }
  }
  return cloud;
}

struct StereoImageProcess::PreProcessResult
StereoImageProcess::ResizeKeepAspectRatio(
    const cv::Mat &input, const cv::Size &target_size) {
  cv::Mat output;
  double scale = std::min((double)target_size.width / input.cols,
                          (double)target_size.height / input.rows);
  // 计算新的尺寸
  cv::Size newSize(static_cast<int>(input.cols * scale),
                   static_cast<int>(input.rows * scale));
  // 调整图像大小
  cv::resize(input, output, newSize, 0, 0, cv::INTER_LINEAR);

  return PreProcessResult{input, output, scale};
}
