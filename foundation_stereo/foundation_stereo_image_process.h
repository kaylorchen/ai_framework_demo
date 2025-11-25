//
// Created by kaylor on 11/25/25.
//

#pragma once
#include "ai_framework.h"
#include "opencv2/opencv.hpp"

class FoundationStereoImageProcess {
public:
  FoundationStereoImageProcess() = delete;
  FoundationStereoImageProcess(const ai_framework::Config &config);
  ~FoundationStereoImageProcess() = default;
  void PreProcess(const std::vector<cv::Mat> &imgs, void **&tensors);
  cv::Mat PostProcess(void **&tensors);

private:
  int width_;
  int height_;
};
