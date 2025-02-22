//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "opencv2/opencv.hpp"
#include "string"
class VideoFile {
 public:
  VideoFile(const std::string &filename);
  ~VideoFile();
  void Show(const float framerate = 25.0);
  cv::Mat GetNextFrame();
  int get_frame_width();
  int get_frame_height();

 private:
  std::string filename_;
  cv::VideoCapture *capture_{nullptr};
};
