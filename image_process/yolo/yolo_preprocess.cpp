//
// Created by kaylor on 6/14/24.
//

#include "yolo_preprocess.h"

YoloPreProcess::YoloPreProcess(int target_side_length, bool debug)
    : target_side_length_(target_side_length), debug_(debug) {}

void YoloPreProcess::Run(const std::vector<cv::Mat> &input, void *tensors[]) {
  for (int i = 0; i < input.size(); ++i) {
    auto &original_input = input.at(i);
    int resize_width = target_side_length_;
    int resize_height = target_side_length_;
    if (original_input.cols >= original_input.rows) {
      float scale = 1.0f * original_input.cols / target_side_length_;
      resize_height = original_input.rows / scale;
    } else {
      float scale = 1.0f * original_input.rows / target_side_length_;
      resize_width = original_input.cols / scale;
    }
    cv::Mat dst;
    cv::resize(original_input, dst, cv::Size(resize_width, resize_height),
               cv::INTER_NEAREST);
#ifdef RK3588
    cv::Mat res = cv::Mat(target_side_length_, target_side_length_, dst.type(),
                          (void *)tensors[i]);
    cv::Mat cvt;
    cv::cvtColor(dst, cvt, cv::COLOR_BGR2RGB);
    MakeSquare(cvt, res);
#else
    cv::Mat res;
    MakeSquare(dst, res);
    PopulateData(res, reinterpret_cast<float *>(tensors[i]));
#endif
    if (debug_) {
      cv::imshow("PreProcess Image", dst);
      cv::waitKey(1);
    }
  }
}

void YoloPreProcess::MakeSquare(const cv::Mat &src, cv::Mat &dst) {
  // 获取图像的宽和高
  int width = src.cols;
  int height = src.rows;
  // 计算需要填充的尺寸
  int border_left = 0;
  int border_right = 0;
  int border_top = 0;
  int border_bottom = 0;

  if (height > width) {
    int delta = height - width;
    border_left += (delta >> 1);
    border_right = border_left;
  } else {
    int delta = width - height;
    border_top += (delta >> 1);
    border_bottom = border_top;
  }
  // 使用灰色(114,114,114)填充边缘
  cv::copyMakeBorder(src, dst, border_top, border_bottom, border_left,
                     border_right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
}

uint64_t YoloPreProcess::PopulateData(const cv::Mat &data, float *dst) {
  if (data.channels() != 3 || data.type() != CV_8UC3) {
    return 0;
  }
  auto *R = dst;
  auto *G = dst + data.total();
  auto *B = dst + data.total() * 2;
  for (int i = 0; i < data.rows; ++i) {
    for (int j = 0; j < data.cols; ++j) {
      // Mat 的数据是BGR
      *B = data.at<cv::Vec3b>(i, j)[0] / 255.0f;
      B++;
      *G = data.at<cv::Vec3b>(i, j)[1] / 255.0f;
      G++;
      *R = data.at<cv::Vec3b>(i, j)[2] / 255.0f;
      R++;
    }
  }
  // 返回填充的字节数
  return data.total() * data.channels() * sizeof(*dst);
}
