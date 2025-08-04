//
// Created by kaylor on 1/9/25.
//

#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H
#include "linux/fb.h"
#include "opencv2/opencv.hpp"
#include "string"

class FrameBuffer {
 public:
  FrameBuffer(std::string device);
  ~FrameBuffer();
  bool WriteFrameBuffer(const cv::Mat image);

 private:
  std::string device_;
  int fd_;
  char *fb_ptr_{nullptr};
  fb_fix_screeninfo fix_screeninfo_;
  fb_var_screeninfo var_screeninfo_;
};

#endif  // FRAMEBUFFER_H
