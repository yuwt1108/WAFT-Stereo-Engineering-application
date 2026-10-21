#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

namespace waft_stereo {
void saveDepth(const std::filesystem::path &folder, const cv::Mat &depth);
void showDepth(const std::filesystem::path &folder, cv::Size original_size);
} // namespace waft_stereo
