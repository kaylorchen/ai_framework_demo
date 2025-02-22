//
// Created by kaylor on 6/15/24.
//

#include "tools.h"

#include "BYTETracker.h"
#include "fstream"
#include "kaylordut/log/logger.h"

void CoordinateTransformation(float &x, float &y, int width, int height,
                              int target_side_length) {
  if (width > height) {
    int padding = (width - height) >> 1;
    float scale = 1.0f * target_side_length / width;
    y = std::max<float>(y / scale - padding, 0);
    x = std::max<float>(x / scale, 0);
  } else {
    int padding = (height - width) >> 1;
    float scale = 1.0f * target_side_length / height;
    x = std::max<float>(x / scale - padding, 0);
    y = std::max<float>(y / scale, 0);
  }
}

void drawSkeleton(cv::Mat &img, const std::vector<cv::Point> &points,
                  const std::vector<int> &pairs, const cv::Scalar &color,
                  int thickness) {
  for (size_t i = 0; i < pairs.size(); i += 2) {
    int index1 = pairs[i];
    int index2 = pairs[i + 1];
    if (points[index1].x != -1 && points[index1].y != -1 &&
        points[index2].x != -1 && points[index2].y != -1) {
      cv::line(img, points[index1], points[index2], color, thickness);
    }
  }
}

void ProcessPoseImage(cv::Mat &image, YoloPostProcess::Result &result,
                      const int target_side_length) {
  std::vector<cv::Point> points(17);
  for (int j = 0; j < 17; ++j) {
    if (result.key_points[j].visibility <= 0.6) {
      points.at(j) = cv::Point(-1, -1);
      continue;
    }
    auto x = result.key_points[j].x;
    auto y = result.key_points[j].y;
    CoordinateTransformation(x, y, image.cols, image.rows, target_side_length);
    points.at(j) = (cv::Point(x, y));
    cv::circle(image, points.at(j), 10, cv::Scalar(0, 0, 255), cv::FILLED,
               cv::LINE_AA);
  }
  std::vector<int> pairs = {
      0,  1,   // Nose to left eye
      1,  3,   // Left eye to left ear
      0,  2,   // Nose to right eye
      2,  4,   // Right eye to right ear
      0,  5,   // Nose to left shoulder
      5,  7,   // Left shoulder to left elbow
      7,  9,   // Left elbow to left wrist
      0,  6,   // Nose to right shoulder
      6,  8,   // Right shoulder to right elbow
      8,  10,  // Right elbow to right wrist
      5,  6,   // Left shoulder to right shoulder
      11, 12,  // Left hip to right hip
      11, 5,   // Left hip to left shoulder
      12, 6,   // Right hip to right shoulder
      11, 13,  // Left hip to left knee
      12, 14,  // Right hip to right knee
      13, 15,  // Left knee to left ankle
      14, 16   // Right knee to right ankle
  };
  drawSkeleton(image, points, pairs, cv::Scalar(255, 0, 0), 2);
}

class KaylordutVideoWriter {
 public:
  KaylordutVideoWriter(std::string filename, int width, int height, float fps) {
    writer_ =
        cv::VideoWriter(filename, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                        fps, cv::Size(width, height), true);
    if (!writer_.isOpened()) {
      KAYLORDUT_LOG_ERROR("Cannot create video writer");
      exit(EXIT_FAILURE);
    }
  }
  ~KaylordutVideoWriter() {
    if (writer_.isOpened()) {
      writer_.release();
    }
  }
  void write(const cv::Mat &image) { writer_.write(image); }

 private:
  cv::VideoWriter writer_;
};
std::shared_ptr<KaylordutVideoWriter> gVideoWriter;

#define N_CLASS_COLORS (20)
unsigned char class_colors[][3] = {
    {255, 56, 56},    // 'FF3838'
    {255, 157, 151},  // 'FF9D97'
    {255, 112, 31},   // 'FF701F'
    {255, 178, 29},   // 'FFB21D'
    {207, 210, 49},   // 'CFD231'
    {72, 249, 10},    // '48F90A'
    {146, 204, 23},   // '92CC17'
    {61, 219, 134},   // '3DDB86'
    {26, 147, 52},    // '1A9334'
    {0, 212, 187},    // '00D4BB'
    {44, 153, 168},   // '2C99A8'
    {0, 194, 255},    // '00C2FF'
    {52, 69, 147},    // '344593'
    {100, 115, 255},  // '6473FF'
    {0, 24, 236},     // '0018EC'
    {132, 56, 255},   // '8438FF'
    {82, 0, 133},     // '520085'
    {203, 56, 255},   // 'CB38FF'
    {255, 149, 200},  // 'FF95C8'
    {255, 55, 199}    // 'FF37C7'
};

// 函数定义：上色函数
// 参数：
//   image - 待修改的图像
//   mask - 掩码图像
//   color - 要应用的颜色（B, G, R）
void ApplyColorWithMask(cv::Mat &image, const cv::Mat &mask,
                        const cv::Scalar &color) {
  // 首先检查图像和掩码的尺寸是否一致
  if (image.size() != mask.size()) {
    throw std::runtime_error("Image and mask sizes do not match");
  }
  // 检查image是否为彩色图像
  if (image.channels() != 3) {
    throw std::runtime_error("Image must be a color image");
  }
  // 遍历图像中的每一个像素
  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {
      // 检查掩码中对应的像素值是否不为0
      if (mask.at<uchar>(y, x) != 0) {
        // 若掩码对应位置不为零，设置图像该位置的颜色
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
      }
    }
  }
}

void AddWeightedSegment(cv::Mat &image, const cv::Mat &seg_mat, int id) {
  auto width = image.cols;
  auto height = image.rows;
  auto seg_width = seg_mat.cols;
  auto seg_height = seg_mat.rows;
  int x = 0, y = 0, rect_w = seg_width, rect_h = seg_height;
  if (width > height) {
    auto padding =
        static_cast<int>((float)(width - height) / width * seg_height / 2);
    KAYLORDUT_LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
    y = std::max(padding - 1, 0);
    rect_h = static_cast<int>((float)height / width * seg_height);
    if (y + rect_h > seg_height) {
      KAYLORDUT_LOG_ERROR("y + rect_h > seg_height");
      exit(EXIT_FAILURE);
    }
  } else {
    auto padding =
        static_cast<int>((float)(height - width) / height * seg_width / 2);
    KAYLORDUT_LOG_WARN_EXPRESSION(padding < 1, "padding is {}", padding);
    x = std::max(padding - 1, 0);
    rect_w = static_cast<int>((float)width / height * seg_width);
    if (x + rect_w > seg_width) {
      KAYLORDUT_LOG_ERROR("x + rect_w > seg_width");
      exit(EXIT_FAILURE);
    }
  }
  cv::Mat all_size_seg_mat;
  cv::resize(seg_mat(cv::Rect(x, y, rect_w, rect_h)), all_size_seg_mat,
             image.size(), cv::INTER_NEAREST);
  auto count = id % N_CLASS_COLORS;
  cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
  ApplyColorWithMask(mask, all_size_seg_mat,
                     cv::Scalar(class_colors[count][0], class_colors[count][1],
                                class_colors[count][2]));
  cv::addWeighted(image, 0.8, mask, 0.2, 0, image);
}

cv::Mat GetImageResult(const cv::Mat &original_image,
                       const int target_side_length,
                       const std::vector<YoloPostProcess::Result> &results,
                       std::vector<std::string> &labels, bool enable_track) {
  auto image = original_image.clone();
  int width = image.cols;
  int height = image.rows;
  std::vector<Object> objects;
  for (auto result : results) {
    AddWeightedSegment(image, result.seg_mat, result.class_id);
    float x1, y1, x2, y2;
    x1 = std::max<float>(result.box.x1, 0);
    y1 = std::max<float>(result.box.y1, 0);
    x2 = std::max<float>(result.box.x2, 0);
    y2 = std::max<float>(result.box.y2, 0);
    CoordinateTransformation(x1, y1, width, height, target_side_length);
    CoordinateTransformation(x2, y2, width, height, target_side_length);
    if (!enable_track) {
      cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2);
      char text[256];
      sprintf(text, "%s %.1f%%", labels.at(result.class_id).c_str(),
              result.obj_prob * 100);
      cv::putText(image, text, cv::Point(x1, y1 + 20), cv::FONT_HERSHEY_COMPLEX,
                  0.65, cv::Scalar(255, 255, 0), 1, cv::LINE_8);
      if (result.model_type == YoloPostProcess::ModelType::POSE_V8) {
        ProcessPoseImage(image, result, target_side_length);
      }
    } else {
      Object object;
      object.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
      object.label = result.class_id;
      object.prob = result.obj_prob;
      objects.push_back(object);
    }
  }
  if (enable_track) {
    static std::unique_ptr<BYTETracker> tracker = nullptr;
    if (tracker == nullptr) {
      tracker = std::make_unique<BYTETracker>(25, 25);
    }
    std::vector<STrack> output_stracks = tracker->update(objects);
    for (size_t i = 0; i < output_stracks.size(); ++i) {
      std::vector<float> tlwh = output_stracks[i].tlwh;
      bool vertical = tlwh[2] / tlwh[3] > 1.6;
      if (tlwh[2] * tlwh[3] > 20 && !vertical) {
        Scalar s = tracker->get_color(output_stracks.at(i).track_id);
        putText(image, format("%d,", output_stracks.at(i).track_id),
                Point(tlwh[0], tlwh[1] - 5), 0, 0.6, Scalar(0, 0, 255), 2,
                LINE_AA);
        rectangle(image, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
      }
    }
  }
  return image;
}
void ShowAndSave(const cv::Mat &image, int cv_wait_ms, bool is_save) {
  cv::imshow("Result", image);
  waitKey(cv_wait_ms);
  if (is_save) {
    if (gVideoWriter == nullptr) {
      auto now = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);
      stringstream ss;
      ss << put_time(localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
      string filename = "result_" + ss.str() + ".mp4";
      gVideoWriter = make_shared<KaylordutVideoWriter>(filename, image.cols,
                                                       image.rows, 25.0);
    }
    gVideoWriter->write(image);
  }
}
void ShowResults(const cv::Mat &original_image, const int target_side_length,
                 const std::vector<YoloPostProcess::Result> &results,
                 std::vector<std::string> &labels, int cv_wait_ms,
                 bool enable_track, bool is_save) {
  auto image = GetImageResult(original_image, target_side_length, results,
                              labels, enable_track);
  ShowAndSave(image, cv_wait_ms, is_save);
}

std::vector<std::string> ReadLabelsFromTextFile(const std::string &filename) {
  std::ifstream file(filename);
  std::vector<std::string> labels;
  std::string label;
  if (file.is_open()) {
    while (getline(file, label)) {
      labels.emplace_back(label);
    }
    file.close();
  } else {
    KAYLORDUT_LOG_ERROR("Could not open {}", filename);
    exit(EXIT_FAILURE);
  }
  return labels;
}
