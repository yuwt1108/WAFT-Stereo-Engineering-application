#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <utility>

namespace waft_stereo {

struct Rectification {
    cv::Mat left_x, left_y, right_x, right_y;
    cv::Mat intrinsic; // Full resolution rectified intrinsics.
    double baseline_m;
};

Rectification makeRectification(double alpha);
std::pair<cv::Mat, cv::Mat> rectifyFrames(const cv::Mat &left, const cv::Mat &right,
                                          const Rectification &maps);
struct StereoPrediction {
    cv::Mat disparity;   // Pixel disparity at the resized inference resolution.
    cv::Mat uncertainty; // Python get_heatmap(): softmax(info channels)[0].
};
StereoPrediction inferStereo(const cv::Mat &left, const cv::Mat &right, cv::Size inference_size,
                             const std::filesystem::path &model_path);
cv::Mat disparityToDepth(const cv::Mat &disparity, double focal_x, double baseline_m, double scale);

} // namespace waft_stereo
