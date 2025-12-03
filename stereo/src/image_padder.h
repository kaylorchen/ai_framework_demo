//
// Created by kaylor on 11/26/25.
//

#pragma once
#include <opencv2/opencv.hpp>

namespace image_processing {
enum class PaddingMode {
  REPLICATE, // 使用 cv::BORDER_REPLICATE 的填充方式
  ADAPTIVE   // 使用类似 image_pad 的自适应填充方式
};

// Pads images such that dimensions are divisible by specified divisor.
// Supports both symmetric and asymmetric padding modes with border replication.
class ImagePadder {
public:
  // Constructs an ImagePadder with given parameters.
  // @param divis_by Divisor for dimension alignment (default: 32).
  explicit ImagePadder(int divis_by = 32, PaddingMode mode = PaddingMode::REPLICATE);

  // Pads the input image to make dimensions divisible by divis_by.
  // @param image Input image to pad.
  // @param symmetric Whether to use symmetric padding (Sintel mode).
  // @param target_size Target size for padding (if empty, use divis_by alignment).
  // @return Padded image with border replication.
  // @throws std::invalid_argument if input image is empty.
  cv::Mat Pad(const cv::Mat &image, bool symmetric = true,
              cv::Size target_size = cv::Size());

  // Removes padding from a previously padded image.
  // @param padded_image Image that was previously padded by this class.
  // @return Original unpadded image.
  // @throws std::invalid_argument if input image is empty.
  // @throws std::runtime_error if ROI calculation is invalid.
  cv::Mat Unpad(const cv::Mat &padded_image);

  // Accessors
  int divis_by() const { return divis_by_; }
  cv::Size original_size() const { return original_size_; }
  cv::Vec4i padding_values() const { return padding_values_; }

  // Mutators
  void set_divis_by(int divis_by) { divis_by_ = divis_by; }

private:
  // Computes padding values based on image dimensions and current settings.
  // @param image_size Size of the input image.
  // @param symmetric Whether to use symmetric padding.
  // @param target_size Target size for padding.
  void ComputePadding(const cv::Size &image_size, bool symmetric,
                      cv::Size target_size = cv::Size());
  cv::Mat AdaptivePadding(const cv::Mat &image, cv::Size target_size);

  int divis_by_;
  cv::Size original_size_;
  cv::Vec4i padding_values_; // [left, right, top, bottom]
  PaddingMode padding_mode_;
};

} // namespace image_processing
