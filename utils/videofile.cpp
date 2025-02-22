//
// Created by kaylor on 3/4/24.
//

#include "videofile.h"

#include "kaylordut/log/logger.h"

VideoFile::VideoFile(const std::string &filename) : filename_(filename) {
  capture_ = new cv::VideoCapture(filename_);
  if (!capture_->isOpened()) {
    KAYLORDUT_LOG_ERROR("Error opening video file");
    exit(EXIT_FAILURE);
  }
}

VideoFile::~VideoFile() {
  if (capture_ != nullptr) {
    KAYLORDUT_LOG_INFO("Release capture")
    capture_->release();
    delete capture_;
  }
}

void VideoFile::Show(const float framerate) {
  const int delay = 1000 / framerate;
  cv::Mat frame;
  while (true) {
    *capture_ >> frame;
    if (frame.empty()) {
      break;
    }
    cv::imshow("Video", frame);
    if (cv::waitKey(delay) >= 0) {
      break;
    }
  }
  cv::destroyAllWindows();
}

cv::Mat VideoFile::GetNextFrame() {
  cv::Mat frame;
  *capture_ >> frame;
  return frame;
}

int VideoFile::get_frame_width() {
  return capture_->get(cv::CAP_PROP_FRAME_WIDTH);
}

int VideoFile::get_frame_height() {
  return capture_->get(cv::CAP_PROP_FRAME_HEIGHT);
}