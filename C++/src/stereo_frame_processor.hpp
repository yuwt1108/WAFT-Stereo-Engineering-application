#pragma once

#include <opencv2/core.hpp>
#include <filesystem>

namespace waft_stereo {

// Accepts raw 1224x1024 BGR frames and returns a depth map at inference_size in meters.
// Providing output_folder also saves rectified images, disparity and uncertainty previews.
cv::Mat process_frames(const cv::Mat &left_frame, const cv::Mat &right_frame,
                       const std::filesystem::path &output_folder = {},
                       cv::Size inference_size = {612, 512},
                       const std::filesystem::path &model_path = {});

} // namespace waft_stereo
