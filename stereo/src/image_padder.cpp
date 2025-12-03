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

ImagePadder::ImagePadder(int divis_by, bool force_square, PaddingMode mode)
    : divis_by_(divis_by), force_square_(force_square), original_size_(0, 0),
      padding_values_(0, 0, 0, 0) {
  if (divis_by_ <= 0) {
    throw std::invalid_argument("divis_by must be positive");
  }
}

cv::Mat ImagePadder::Pad(const cv::Mat &image, bool symmetric,
                         cv::Size target_size) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  original_size_ = image.size();
  ComputePadding(original_size_, symmetric, target_size);

  cv::Mat padded_image;
  const int top = padding_values_[2];
  const int bottom = padding_values_[3];
  const int left = padding_values_[0];
  const int right = padding_values_[1];

  switch (padding_mode_) {
  case PaddingMode::REPLICATE:
    cv::copyMakeBorder(image, padded_image, top, bottom, left, right,
                       cv::BORDER_REPLICATE);
    break;
  case PaddingMode::ADAPTIVE:
    padded_image = AdaptivePadding(image);
    break;
  default:
    throw std::runtime_error("Unknown padding mode");
  }
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

void ImagePadder::ComputePadding(const cv::Size &image_size, bool symmetric,
                                 cv::Size target_size) {
  const int height = image_size.height;
  const int width = image_size.width;

  int pad_height = 0;
  int pad_width = 0;
  if (target_size.empty()) {
    if (force_square_) {
      const int max_side = std::max(height, width);
      pad_height = ((max_side / divis_by_) + 1) * divis_by_ - height;
      pad_width = ((max_side / divis_by_) + 1) * divis_by_ - width;
    } else {
      pad_height =
          (((height / divis_by_) + 1) * divis_by_ - height) % divis_by_;
      pad_width = (((width / divis_by_) + 1) * divis_by_ - width) % divis_by_;
    }
  } else {
    if (force_square_) {
      auto pad = std::max(target_size.height - image_size.height,
                          target_size.width - image_size.width);
      pad_width = pad;
      pad_height = pad;
    } else {
      pad_height = target_size.height - image_size.height;
      pad_width = target_size.width - image_size.width;
    }
  }
  if (symmetric) {
    // 真正的对称填充：四边均匀分配
    int pad_left = pad_width / 2;
    int pad_right = pad_width - pad_left;
    int pad_top = pad_height / 2;
    int pad_bottom = pad_height - pad_top;

    padding_values_ = cv::Vec4i(pad_left, pad_right, pad_top, pad_bottom);
  } else {
    // 更合理的非对称填充：只在右侧和底部填充
    padding_values_ = cv::Vec4i(0, pad_width, 0, pad_height);
  }
}

cv::Mat ImagePadder::AdaptivePadding(const cv::Mat &image) {
  int H = image.rows;
  int W = image.cols;

  int H_new = ((H + divis_by_ - 1) / divis_by_) * divis_by_;
  int W_new = ((W + divis_by_ - 1) / divis_by_) * divis_by_;

  int pad_h = H_new - H;
  int pad_w = W_new - W;

  // 计算需要填充的尺寸
  int top = pad_h / 2;
  // int bottom = pad_h - top;
  int left = pad_w / 2;
  // int right = pad_w - left;

  // 创建一个足够大的空白图像，并用平均值填充
  cv::Scalar mean_color = cv::mean(image);
  cv::Mat padded_image(H_new, W_new, image.type(), mean_color);

  // 将原图放置在中心位置
  image.copyTo(padded_image(cv::Rect(left, top, W, H)));

  return padded_image;
}

} // namespace image_processing
