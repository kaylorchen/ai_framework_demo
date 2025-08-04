//
// Created by kaylor on 1/9/25.
//

#include "framebuffer.h"

#include <unistd.h>

#include "kaylordut/log/logger.h"
#include "sys/file.h"
#include "sys/ioctl.h"
#include "sys/mman.h"

FrameBuffer::FrameBuffer(std::string device) {
  device_ = device;
  fd_ = open(device_.c_str(), O_RDWR);
  if (fd_ == -1) {
    KAYLORDUT_LOG_ERROR("cannot open {}", device_);
    exit(EXIT_FAILURE);
  }
  // 尝试独占锁定帧缓冲设备
  if (flock(fd_, LOCK_EX | LOCK_NB) == -1) {
    KAYLORDUT_LOG_ERROR("Error locking framebuffer device");
    close(fd_);
    exit(EXIT_FAILURE);
  }

  if (ioctl(fd_, FBIOGET_FSCREENINFO, &fix_screeninfo_)) {
    KAYLORDUT_LOG_ERROR("cannot get fix screen info");
    flock(fd_, LOCK_UN);
    close(fd_);
    exit(EXIT_FAILURE);
  }
  KAYLORDUT_LOG_INFO("fix screen info: smem_len: {}, line_len: {}",
                     fix_screeninfo_.smem_len, fix_screeninfo_.line_length);
  if (ioctl(fd_, FBIOGET_VSCREENINFO, &var_screeninfo_)) {
    KAYLORDUT_LOG_ERROR("cannot var fix screen info");
    flock(fd_, LOCK_UN);
    close(fd_);
    exit(EXIT_FAILURE);
  }
  KAYLORDUT_LOG_INFO("var screen info: {}x{}", var_screeninfo_.xres,
                     var_screeninfo_.yres);
  fb_ptr_ = (char *)mmap(0, fix_screeninfo_.smem_len, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, 0);
  if ((long)fb_ptr_ == -1) {
    KAYLORDUT_LOG_ERROR("map mem error");
    flock(fd_, LOCK_UN);
    close(fd_);
    exit(EXIT_FAILURE);
  }
}

FrameBuffer::~FrameBuffer() {
  if (fb_ptr_ != nullptr) {
    munmap(fb_ptr_, fix_screeninfo_.smem_len);
    if (fd_) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
    fb_ptr_ = nullptr;
  }
}

bool FrameBuffer::WriteFrameBuffer(const cv::Mat image) {
  cv::Mat result;
  cv::resize(image, result,
             cv::Size(var_screeninfo_.xres, var_screeninfo_.yres));
  if (image.channels() != 4) {
    KAYLORDUT_TIME_COST_DEBUG("BGR2BGRA",
                             cv::cvtColor(result, result, cv::COLOR_BGR2BGRA));
  }
  auto dst = fb_ptr_;
  auto src = result.data;
  for (int i = 0; i < result.rows; ++i) {
    std::memcpy(dst, src, result.cols * result.elemSize());
    dst += fix_screeninfo_.line_length;
    src += result.cols * result.elemSize();
  }
  return true;
}
