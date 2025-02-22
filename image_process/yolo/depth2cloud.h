//
// Created by kaylor on 9/29/24.
//

#pragma once

#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

class Depth2Cloud {
 public:
  struct CameraInfo {
    int width, height;
    Eigen::Matrix<double, 5, 1> d;
    Eigen::Matrix3d k;
    int binning_x;
    int binning_y;
  };

  struct Box{
    int x1=0, y1=0, x2=0, y2=0;
  };

  static pcl::PointCloud<pcl::PointXYZ>::Ptr Convert(
      const cv::Mat &input, const Depth2Cloud::CameraInfo &camera_info,
      const Depth2Cloud::Box &box, float depth_scale = 1000.0,
      float scale_x = 1.0, float scale_y = 1.0);

  static pcl::PointCloud<pcl::PointXYZ>::Ptr Convert(
      const cv::Mat &input, const cv::Mat mask,
      const Depth2Cloud::CameraInfo &camera_info, const Depth2Cloud::Box &box,
      float depth_scale = 1000.0, float scale_x = 1.0, float scale_y = 1.0);
};