#include "stereo.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/script.h>

namespace fs = std::filesystem;
namespace waft_stereo {
namespace {

torch::jit::Module loadModel(const fs::path &path) {
    if (!torch::cuda::is_available())
        throw std::runtime_error("CUDA LibTorch/GPU is required by the exported model");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    // Keep eager operation order when switching input shapes.
    torch::jit::setGraphExecutorOptimize(false);
    torch::jit::ExtraFilesMap metadata{{"contract.txt", ""}};
    auto module = torch::jit::load(path.string(), torch::Device(torch::kCUDA), metadata);
    if (metadata["contract.txt"] != "normalized_rgb_symmetric_pad16_dynamic_v1")
        throw std::runtime_error("Model preprocessing contract does not match this C++ program");
    module.eval();
    return module;
}

torch::Tensor modelInput(const cv::Mat &image, int pad_h, int pad_w) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    auto raw = torch::from_blob(rgb.data, {1, rgb.rows, rgb.cols, 3}, torch::kUInt8)
                   .permute({0, 3, 1, 2})
                   .to(torch::kCUDA)
                   .to(torch::kFloat32);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto mean = torch::tensor({0.485f, 0.456f, 0.406f}, options).view({1, 3, 1, 1});
    auto deviation = torch::tensor({0.229f, 0.224f, 0.225f}, options).view({1, 3, 1, 1});
    return at::constant_pad_nd(((raw / 255.0 - mean) / deviation).contiguous(),
                               {pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2}, 0);
}

} // namespace

Rectification makeRectification(double alpha) {
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
        throw std::invalid_argument("Alpha must be between 0 and 1");
    const cv::Size size(1224, 1024);
    // 2026-10-21 underwater calibration; MATLAB stereoParametersToOpenCV export.
    // clang-format off
    cv::Mat left_intrinsic = (cv::Mat_<double>(3, 3) <<
        796.199228041569, 0, 609.47994263749069,
        0, 804.14616192489359, 505.54377526426208,
        0, 0, 1);
    cv::Mat right_intrinsic = (cv::Mat_<double>(3, 3) <<
        794.23055503378964, 0, 603.13176622226558,
        0, 802.873594828302, 497.99463516189127,
        0, 0, 1);
    cv::Mat left_distortion = (cv::Mat_<double>(1, 5) <<
        0.073401780189902635, 0.42856774465131858, 0, 0, 0);
    cv::Mat right_distortion = (cv::Mat_<double>(1, 5) <<
        0.091691292073727837, 0.36722066600693049, 0, 0, 0);
    cv::Mat rotation = (cv::Mat_<double>(3, 3) <<
        0.99999872316149441, -0.00011403992536902048, 0.0015939480155490809,
        0.00011315567111308065, 0.99999983967614259, 0.00055483644732478524,
        -0.0015940110335082292, -0.00055465537463083247, 0.999998575742106);
    cv::Mat translation = (cv::Mat_<double>(3, 1) <<
        -59.723322349645692, 0.010047843490347319, -0.39604673321685346);
    // clang-format on
    cv::Mat left_rotation, right_rotation, left_projection, right_projection, reprojection;
    cv::stereoRectify(left_intrinsic, left_distortion, right_intrinsic, right_distortion, size,
                      rotation, translation, left_rotation, right_rotation, left_projection,
                      right_projection, reprojection, cv::CALIB_ZERO_DISPARITY, alpha);
    Rectification maps;
    maps.intrinsic = left_projection(cv::Rect(0, 0, 3, 3)).clone();
    // Calibration translation is in millimeters; depth uses meters.
    maps.baseline_m =
        std::abs(right_projection.at<double>(0, 3) / right_projection.at<double>(0, 0)) / 1000.0;
    cv::initUndistortRectifyMap(left_intrinsic, left_distortion, left_rotation, maps.intrinsic,
                                size, CV_32FC1, maps.left_x, maps.left_y);
    cv::initUndistortRectifyMap(right_intrinsic, right_distortion, right_rotation,
                                right_projection(cv::Rect(0, 0, 3, 3)), size, CV_32FC1,
                                maps.right_x, maps.right_y);
    return maps;
}

std::pair<cv::Mat, cv::Mat> rectifyFrames(const cv::Mat &left, const cv::Mat &right,
                                          const Rectification &maps) {
    if (left.empty() || right.empty() || left.type() != CV_8UC3 || right.type() != CV_8UC3)
        throw std::invalid_argument("Frames must be nonempty CV_8UC3 BGR images");
    if (left.size() != maps.left_x.size() || right.size() != maps.right_x.size())
        throw std::invalid_argument("Current stereo calibration requires two raw 1224x1024 frames");
    cv::Mat left_rect, right_rect;
    cv::remap(left, left_rect, maps.left_x, maps.left_y, cv::INTER_LINEAR);
    cv::remap(right, right_rect, maps.right_x, maps.right_y, cv::INTER_LINEAR);
    return {left_rect, right_rect};
}

StereoPrediction inferStereo(const cv::Mat &left, const cv::Mat &right, cv::Size inference_size,
                             const fs::path &model_path) {
    if (left.empty() || right.empty() || left.size() != right.size() || left.type() != CV_8UC3 ||
        right.type() != CV_8UC3)
        throw std::invalid_argument("Invalid rectified pair");
    if (inference_size.width < 32 || inference_size.height < 32 ||
        inference_size.width > left.cols || inference_size.height > left.rows)
        throw std::invalid_argument("Inference size must be between 32x32 and the input size");

    cv::Mat scaled_left, scaled_right;
    cv::resize(left, scaled_left, inference_size);
    cv::resize(right, scaled_right, inference_size);
    const auto size = scaled_left.size();

    // Reuse weights across frames; reload only when the requested path changes.
    static auto module = loadModel(model_path);
    static auto loaded_path = model_path;
    if (loaded_path != model_path) {
        module = loadModel(model_path);
        loaded_path = model_path;
    }
    const int pad_h = (16 - size.height % 16) % 16;
    const int pad_w = (16 - size.width % 16) % 16;
    torch::NoGradGuard guard;
    auto left_input = modelInput(scaled_left, pad_h, pad_w);
    auto right_input = modelInput(scaled_right, pad_h, pad_w);
    auto predictions = module.forward({left_input, right_input}).toTuple();
    if (predictions->elements().size() != 2)
        throw std::runtime_error("Model must return disparity and uncertainty logits");
    auto disparity = predictions->elements()[0]
                         .toTensor()
                         .slice(1, pad_h / 2, pad_h / 2 + size.height)
                         .slice(2, pad_w / 2, pad_w / 2 + size.width)
                         .to(torch::kCPU)
                         .to(torch::kFloat32)
                         .contiguous();
    if (disparity.dim() != 3 || disparity.size(0) != 1 || disparity.size(1) != size.height ||
        disparity.size(2) != size.width)
        throw std::runtime_error("Model output shape does not match input");
    auto info = predictions->elements()[1].toTensor();
    if (info.dim() != 4 || info.size(0) != 1 || info.size(1) < 2)
        throw std::runtime_error("Model uncertainty output has an invalid shape");
    auto uncertainty = info.narrow(1, 0, 2).softmax(1).select(1, 0)
                           .slice(1, pad_h / 2, pad_h / 2 + size.height)
                           .slice(2, pad_w / 2, pad_w / 2 + size.width)
                           .to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (uncertainty.dim() != 3 || uncertainty.size(0) != 1 ||
        uncertainty.size(1) != size.height || uncertainty.size(2) != size.width)
        throw std::runtime_error("Model uncertainty shape does not match input");
    // Mat must own its pixels after the temporary Torch tensors are released.
    return {cv::Mat(size, CV_32F, disparity.data_ptr<float>()).clone(),
            cv::Mat(size, CV_32F, uncertainty.data_ptr<float>()).clone()};
}

cv::Mat disparityToDepth(const cv::Mat &disparity, double focal_x, double baseline_m,
                         double scale) {
    if (disparity.empty() || disparity.type() != CV_32FC1)
        throw std::invalid_argument("Disparity must be a nonempty float32 matrix");
    if (!std::isfinite(focal_x) || !std::isfinite(baseline_m) || !std::isfinite(scale) ||
        focal_x <= 0 || baseline_m <= 0 || scale <= 0)
        throw std::invalid_argument("Invalid baseline/focal length/scale");
    cv::Mat depth(disparity.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            const float d = disparity.at<float>(y, x);
            if (!std::isfinite(d) || d <= 0)
                continue;
            const float value = static_cast<float>(focal_x * scale * baseline_m / d);
            if (std::isfinite(value) && value > 0)
                depth.at<float>(y, x) = value;
        }
    }
    return depth;
}

} // namespace waft_stereo
