//
// Created by kaylor on 11/25/25.
//

#include "foundation_stereo_image_process.h"

FoundationStereoImageProcess::FoundationStereoImageProcess(
    const ai_framework::Config &config) {
  width_ = config.input_layer_shape.at("left").at(3);
  height_ = config.input_layer_shape.at("left").at(2);
}

void FoundationStereoImageProcess::PreProcess(const std::vector<cv::Mat> &imgs,
                                              void **&tensors) {
  for (long unsigned int i = 0; i < imgs.size(); ++i) {
    cv::Mat img = imgs.at(i);
    cv::resize(img, img, cv::Size(width_, height_));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    auto dst = reinterpret_cast<float *>(tensors[i]);
    auto *R = dst;
    auto *G = dst + img.total();
    auto *B = dst + img.total() * 2;
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        // Mat 的数据是BGR
        *B = img.at<cv::Vec3f>(i, j)[0];
        B++;
        *G = img.at<cv::Vec3f>(i, j)[1];
        G++;
        *R = img.at<cv::Vec3f>(i, j)[2];
        R++;
      }
    }
  }
}

cv::Mat FoundationStereoImageProcess::PostProcess(void **&tensors) {
  cv::Mat disparity_map(height_, width_, CV_32FC1, tensors[0]);
  cv::Mat depth_normalized;
  cv::normalize(disparity_map, depth_normalized, 0, 255, cv::NORM_MINMAX,
                CV_8UC1);
  cv::Mat depth_colored;
  cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);
  return depth_colored;
}
