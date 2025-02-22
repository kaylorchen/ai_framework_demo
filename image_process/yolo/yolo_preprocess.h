//
// Created by kaylor on 6/14/24.
//

#pragma once
#include "opencv2/opencv.hpp"

class YoloPreProcess {
 public:
  YoloPreProcess() = delete;
  YoloPreProcess(int target_side_length, bool debug = false);
  void Run(const std::vector<cv::Mat> &input, void *tensors[]);
  const int &get_target_side_length() const { return target_side_length_; }

 private:
  void MakeSquare(const cv::Mat &src, cv::Mat &dst);
  uint64_t PopulateData(const cv::Mat &data, float *dst);
  int target_side_length_;
  bool debug_ = {false};
};
