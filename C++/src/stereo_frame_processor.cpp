#include "stereo_frame_processor.hpp"
#include "stereo.hpp"
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat waft_stereo::process_frames(const cv::Mat &left_frame, const cv::Mat &right_frame,
                                   const std::filesystem::path &output_folder,
                                   cv::Size inference_size,
                                   const std::filesystem::path &model_path) {
    static const auto maps = makeRectification(0.2);
    auto pair = rectifyFrames(left_frame, right_frame, maps);
    const auto path = model_path.empty()
                          ? std::filesystem::path(WAFT_PROJECT_DIR) / "data/model/waft_B.pt"
                          : model_path;
    auto prediction = inferStereo(pair.first, pair.second, inference_size, path);
    const cv::Mat &disparity = prediction.disparity;
    const double focal_x = maps.intrinsic.at<double>(0, 0);
    // Disparity is horizontal, so only the horizontal resize ratio scales focal length.
    const double horizontal_scale = static_cast<double>(inference_size.width) / pair.first.cols;
    cv::Mat depth = disparityToDepth(disparity, focal_x, maps.baseline_m, horizontal_scale);
    if (!output_folder.empty()) {
        std::filesystem::create_directories(output_folder);
        cv::Mat alignment;
        cv::hconcat(pair.first, pair.second, alignment);
        for (int y = 0; y < alignment.rows; y += 100)
            cv::line(alignment, {0, y}, {alignment.cols - 1, y}, {0, 255, 0}, 1);

        // Normalize valid disparities only; this colored image is for comparison, not depth conversion.
        cv::Mat valid = (disparity > 0) & (disparity < std::numeric_limits<float>::infinity());
        cv::Mat gray(disparity.size(), CV_8U, cv::Scalar(0)), relative_disparity;
        if (cv::countNonZero(valid) > 0) {
            double low, high;
            cv::minMaxLoc(disparity, &low, &high, nullptr, nullptr, valid);
            if (high > low)
                disparity.convertTo(gray, CV_8U, 255.0 / (high - low), -low * 255.0 / (high - low));
        }
        cv::applyColorMap(gray, relative_disparity, cv::COLORMAP_TURBO);
        relative_disparity.setTo(cv::Scalar(0, 0, 0), ~valid);
        cv::Mat uncertainty_gray, uncertainty_color, uncertainty_background, uncertainty_vis;
        prediction.uncertainty.convertTo(uncertainty_gray, CV_8U, 255.0);
        cv::applyColorMap(uncertainty_gray, uncertainty_color, cv::COLORMAP_JET);
        cv::resize(pair.first, uncertainty_background, disparity.size());
        cv::addWeighted(uncertainty_background, 0.3, uncertainty_color, 0.7, 0,
                        uncertainty_vis);
        cv::Mat gradient(50, uncertainty_vis.cols, CV_8U);
        for (int x = 0; x < gradient.cols; ++x)
            gradient.col(x).setTo(cv::Scalar(255.0 * x / (gradient.cols - 1)));
        cv::Mat color_bar;
        cv::applyColorMap(gradient, color_bar, cv::COLORMAP_JET);
        cv::vconcat(uncertainty_vis, color_bar, uncertainty_vis);
        for (const auto &item : {
                 std::make_pair("left_rect.png", pair.first),
                 std::make_pair("right_rect.png", pair.second),
                 std::make_pair("rectified_preview.png", alignment),
                 std::make_pair("relative_disparity.png", relative_disparity),
                 std::make_pair("uncertainty.png", uncertainty_vis)}) {
            const auto path = output_folder / item.first;
            if (!cv::imwrite(path.string(), item.second))
                throw std::runtime_error("Cannot write " + path.string());
        }
    }
    return depth;
}
