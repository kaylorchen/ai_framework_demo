//
// Created by kaylor on 9/29/24.
//

#include "depth2cloud.h"

#include "kaylordut/log/logger.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2Cloud::Convert(
    const cv::Mat &input, const cv::Mat mask,
    const Depth2Cloud::CameraInfo &camera_info, const Depth2Cloud::Box &box,
    float depth_scale, float scale_x, float scale_y) {
  auto width = input.cols;
  auto height = input.rows;
  auto seg_width = mask.cols;
  auto seg_height = mask.rows;
  int x = 0, y = 0, rect_w = seg_width, rect_h = seg_height;
  if (width > height) {
    auto padding =
        static_cast<int>((float)(width - height) / width * seg_height / 2);
    KAYLORDUT_LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
    y = std::max(padding - 1, 0);
    rect_h = static_cast<int>((float)height / width * seg_height);
    if (y + rect_h > seg_height) {
      KAYLORDUT_LOG_ERROR("y + rect_h > seg_height");
      exit(EXIT_FAILURE);
    }
  } else {
    auto padding =
        static_cast<int>((float)(height - width) / height * seg_width / 2);
    KAYLORDUT_LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
    x = std::max(padding - 1, 0);
    rect_w = static_cast<int>((float)width / height * seg_width);
    if (x + rect_w > seg_width) {
      KAYLORDUT_LOG_ERROR("x + rect_w > seg_width");
      exit(EXIT_FAILURE);
    }
  }
  cv::Mat all_size_mask;
  cv::resize(mask(cv::Rect(x, y, rect_w, rect_h)), all_size_mask, input.size(),
             cv::INTER_NEAREST);
  cv::Mat masked_image = cv::Mat::zeros(input.size(), input.type());
  input.copyTo(masked_image, all_size_mask);
  return Convert(masked_image, camera_info, box, depth_scale, scale_x, scale_y);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2Cloud::Convert(const cv::Mat &input,
                                                         const Depth2Cloud::CameraInfo &camera_info,
                                                         const Depth2Cloud::Box &box,
                                                         float depth_scale,
                                                         float scale_x,
                                                         float scale_y) {
  int resized_width = input.cols * scale_x;
  int resized_height = input.rows * scale_y;
  auto fx = camera_info.k(0, 0) * scale_x;
  auto fy = camera_info.k(1, 1) * scale_y;
  auto cx = camera_info.k(0, 2) * scale_x;
  auto cy = camera_info.k(1, 2) * scale_y;
  cv::Mat depth_image;
  Box vaild_box = {0};
  if (resized_height == input.rows && resized_width == input.cols){
    depth_image = input;
    vaild_box = box;
  }else{
    cv::resize(input, depth_image, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_NEAREST);
    vaild_box.x1 = box.x1 * scale_x;
    vaild_box.y1 = box.y1 * scale_y;
    vaild_box.x2 = box.x2 * scale_x;
    vaild_box.y2 = box.y2 * scale_y;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  float depth = 0;
  int row_num = std::abs(vaild_box.y2 - vaild_box.y1);
  int col_num = std::abs(vaild_box.x2 - vaild_box.x1);
  (cloud->points).reserve(row_num * col_num);
  for (int row = vaild_box.y1; row < vaild_box.y2; row++) {
    for (int col = vaild_box.x1; col < vaild_box.x2; col++) {
      depth = depth_image.at<uint16_t>(row, col) / depth_scale;
      pcl::PointXYZ point(0.0, 0.0, 0.0);
      if ((depth > 0.000) && (depth < 3.000)) {
        point.z = depth;
        point.x = (col - cx) * point.z / fx;
        point.y = (row - cy) * point.z / fy;
      }
      cloud->emplace_back(std::move(point));
    }
  }
  return cloud;
}