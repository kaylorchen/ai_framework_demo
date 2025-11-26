//
// Created by kaylor on 11/25/25.
//

#include "foundation_stereo_image_process.h"

#include <kaylordut/log/logger.h>

FoundationStereoImageProcess::FoundationStereoImageProcess(
    const ai_framework::Config &config, const std::vector<float> &K,
    float baseline) {
  width_ = config.input_layer_shape.at("left").at(3);
  height_ = config.input_layer_shape.at("left").at(2);
  K_ = K;
  baseline_ = baseline;
}

void FoundationStereoImageProcess::PreProcess(const std::vector<cv::Mat> &imgs,
                                              void **&tensors) {
  for (long unsigned int i = 0; i < imgs.size(); ++i) {
    cv::Mat img_padded = padder_.Pad(imgs[i]);
    cv::Mat img_resized;
    if (img_padded.size() != cv::Size(width_, height_)) {
      KAYLORDUT_LOG_INFO("img_padded_size = {}x{} --> target_size = {}x{}",
                         img_padded.cols, img_padded.rows, width_, height_);
      cv::resize(img_padded, img_resized, cv::Size(width_, height_));
    } else {
      img_resized = img_padded;
    }

    // 转换为float
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32FC3, 1.0f);

    // 分离通道（OpenCV优化过的操作）
    std::vector<cv::Mat> bgr_channels;
    cv::split(img_float, bgr_channels);

    auto dst = reinterpret_cast<float *>(tensors[i]);
    const int total_pixels = img_float.total();

    // 注意：bgr_channels[0]=B, [1]=G, [2]=R
    // 我们需要RGB顺序，所以bgr_channels[2]是R，[1]是G，[0]是B

    // 使用memcpy进行批量复制（最高效）
    memcpy(dst, bgr_channels[2].data, total_pixels * sizeof(float)); // R
    memcpy(dst + total_pixels, bgr_channels[1].data,
           total_pixels * sizeof(float)); // G
    memcpy(dst + total_pixels * 2, bgr_channels[0].data,
           total_pixels * sizeof(float)); // B

    // cv::Mat img = imgs.at(i);
    // cv::resize(img, img, cv::Size(width_, height_));
    // img.convertTo(img, CV_32FC3, 1.0f);
    //
    // auto dst = reinterpret_cast<float *>(tensors[i]);
    // auto *R = dst;
    // auto *G = dst + img.total();
    // auto *B = dst + img.total() * 2;
    // for (int i = 0; i < img.rows; ++i) {
    //   for (int j = 0; j < img.cols; ++j) {
    //     // Mat 的数据是BGR
    //     *B = img.at<cv::Vec3f>(i, j)[0];
    //     B++;
    //     *G = img.at<cv::Vec3f>(i, j)[1];
    //     G++;
    //     *R = img.at<cv::Vec3f>(i, j)[2];
    //     R++;
    //   }
    // }
  }
}

std::shared_ptr<FoundationStereoImageProcess::ProcessResult>
FoundationStereoImageProcess::PostProcess(void **&tensors,
                                          const cv::Mat &original_img) {
  auto result = std::make_shared<ProcessResult>();
  result->original_img = original_img;
  auto inference_output = cv::Mat(height_, width_, CV_32FC1, tensors[0]);
  KAYLORDUT_LOG_INFO("inference_output_size = {}x{}", inference_output.cols,
                     inference_output.rows);
  result->disparity_img = padder_.Unpad(inference_output);
  KAYLORDUT_LOG_INFO("disparity_img_size = {}x{}", result->disparity_img.cols,
                     result->disparity_img.rows);
  RemoveInvisiblePoints(result->disparity_img);
  result->depth_img = cv::Mat(result->disparity_img.size(), CV_32FC1);
  ComputeDepth(result->depth_img, result->disparity_img, K_[0], baseline_);
  result->cloud = DepthImageToPointCloud(result->depth_img, original_img, K_);

  // cv::Mat depth_normalized;
  // cv::normalize(disparity_map, depth_normalized, 0, 255, cv::NORM_MINMAX,
  //               CV_8UC1);
  // cv::Mat depth_colored;
  // cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);
  return result;
}

void FoundationStereoImageProcess::RemoveInvisiblePoints(cv::Mat &disp) {
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

void FoundationStereoImageProcess::ComputeDepth(cv::Mat &depth,
                                                const cv::Mat &disp, float K00,
                                                float baseline) {
  if (disp.empty())
    return;
  // 预计算常量
  float scale = K00 * baseline;
  cv::parallel_for_(cv::Range(0, disp.rows), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; y++) {
      const float *disp_ptr = disp.ptr<float>(y);
      float *depth_ptr = depth.ptr<float>(y);

      for (int x = 0; x < disp.cols; x++) {
        float d = disp_ptr[x];
        // 避免除以零和无效视差
        if (d > 0 && std::isfinite(d)) {
          depth_ptr[x] = scale / d;
        } else {
          depth_ptr[x] = 0.0f; // 或者设为其他无效值
        }
      }
    }
  });
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
FoundationStereoImageProcess::DepthImageToPointCloud(const cv::Mat &depth_img,
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
