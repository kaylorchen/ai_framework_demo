//
// Created by kaylor on 11/25/25.
//

#pragma once
#include "ai_framework.h"
#include "image_padder.h"
#include "opencv2/opencv.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class FoundationStereoImageProcess {
public:
  struct PostProcessResult {
    cv::Mat original_img;
    cv::Mat disparity_img;
    cv::Mat depth_img;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  };

  struct PreProcessResult {
    cv::Mat original_img;
    cv::Mat resized_img;
    double scale;
  };

  FoundationStereoImageProcess() = delete;
  FoundationStereoImageProcess(const ai_framework::Config &config,
                               const std::vector<float> &K, float baseline);
  ~FoundationStereoImageProcess() = default;
  std::shared_ptr<PreProcessResult> PreProcess(const std::vector<cv::Mat> &imgs,
                                               void **&tensors);
  std::shared_ptr<PostProcessResult>
  PostProcess(void **&tensors, const PreProcessResult &pre_process_result);

private:
  int width_;
  int height_;
  std::vector<float> K_;
  float baseline_;
  image_processing::ImagePadder padder_;
  void RemoveInvisiblePoints(cv::Mat &disp);
  void ComputeDepth(cv::Mat &depth, const cv::Mat &disp, float K00,
                    float baseline);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  DepthImageToPointCloud(const cv::Mat &depth, const cv::Mat &rgb,
                         std::vector<float> &K);
  struct PreProcessResult ResizeKeepAspectRatio(const cv::Mat &input,
                                                const cv::Size &target_size);
};
