//
// Created by kaylor on 6/15/24.
//

#pragma once
#include "image_process/yolo/yolo_postprocess.h"
#include "opencv2/opencv.hpp"
void ShowResults(const cv::Mat &original_image, const int target_side_length,
                 const std::vector<YoloPostProcess::Result> &results,
                 std::vector<std::string> &labels, int cv_wait_ms,
                 bool enable_track = false, bool is_save = false);
void ShowAndSave(const cv::Mat &image, int cv_wait_ms, bool is_save);
cv::Mat GetImageResult(const cv::Mat &original_image,
                       const int target_side_length,
                       const std::vector<YoloPostProcess::Result> &results,
                       std::vector<std::string> &labels,
                       bool enable_track = false);

void AddWeightedSegment(cv::Mat &image, const cv::Mat &seg_mat, int id);

std::vector<std::string> ReadLabelsFromTextFile(const std::string &filename);
