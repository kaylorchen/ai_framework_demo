//
// Created by kaylor on 11/26/25.
//

#include "image_padder.h"

#include <stdexcept>

namespace image_processing {

namespace {

constexpr int kDefaultDivisBy = 32;
constexpr bool kDefaultForceSquare = false;

} // namespace

ImagePadder::ImagePadder(int divis_by, bool force_square)
    : divis_by_(divis_by), force_square_(force_square), original_size_(0, 0),
      padding_values_(0, 0, 0, 0) {
  if (divis_by_ <= 0) {
    throw std::invalid_argument("divis_by must be positive");
  }
}

cv::Mat ImagePadder::Pad(const cv::Mat &image, bool symmetric) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  original_size_ = image.size();
  ComputePadding(original_size_, symmetric);

  cv::Mat padded_image;
  const int top = padding_values_[2];
  const int bottom = padding_values_[3];
  const int left = padding_values_[0];
  const int right = padding_values_[1];

  cv::copyMakeBorder(image, padded_image, top, bottom, left, right,
                     cv::BORDER_REPLICATE);

  return padded_image;
}

cv::Mat ImagePadder::Unpad(const cv::Mat &padded_image) {
  if (padded_image.empty()) {
    throw std::invalid_argument("Padded image is empty");
  }

  const int height = padded_image.rows;
  const int width = padded_image.cols;
  const int left = padding_values_[0];
  const int right = padding_values_[1];
  const int top = padding_values_[2];
  const int bottom = padding_values_[3];

  const int roi_width = width - left - right;
  const int roi_height = height - top - bottom;

  // Validate ROI bounds
  if (roi_width <= 0 || roi_height <= 0) {
    throw std::runtime_error("Invalid ROI: dimensions non-positive");
  }
  if (left < 0 || top < 0 || left + roi_width > width ||
      top + roi_height > height) {
    throw std::runtime_error("Invalid ROI: out of bounds");
  }

  const cv::Rect roi(left, top, roi_width, roi_height);
  return padded_image(roi).clone();
}

void ImagePadder::ComputePadding(const cv::Size &image_size, bool symmetric) {
  const int height = image_size.height;
  const int width = image_size.width;

  int pad_height = 0;
  int pad_width = 0;

  if (force_square_) {
    const int max_side = std::max(height, width);
    pad_height = ((max_side / divis_by_) + 1) * divis_by_ - height;
    pad_width = ((max_side / divis_by_) + 1) * divis_by_ - width;
  } else {
    pad_height = (((height / divis_by_) + 1) * divis_by_ - height) % divis_by_;
    pad_width = (((width / divis_by_) + 1) * divis_by_ - width) % divis_by_;
  }

  if (symmetric) {
    // Sintel mode: symmetric padding (top/bottom, left/right)
    padding_values_ = cv::Vec4i(pad_width / 2,                // left
                                pad_width - pad_width / 2,    // right
                                pad_height / 2,               // top
                                pad_height - pad_height / 2); // bottom
  } else {
    // Asymmetric mode: pad only bottom and right
    padding_values_ = cv::Vec4i(pad_width / 2,             // left
                                pad_width - pad_width / 2, // right
                                0,                         // top
                                pad_height);               // bottom
  }
}

} // namespace image_processing
