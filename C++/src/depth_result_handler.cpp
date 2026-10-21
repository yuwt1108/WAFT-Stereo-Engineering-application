#include "depth_result_handler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace waft_stereo {
void writeImage(const fs::path &path, const cv::Mat &image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Cannot write " + path.string());
}

// NPY v1/v2, C-order, little-endian float32: the numerical format used by this pipeline.
void writeNpy(const fs::path &path, const cv::Mat &matrix) {
    if (matrix.type() != CV_32FC1 || matrix.empty())
        throw std::runtime_error("Expected nonempty float32 matrix");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(matrix.rows) + ", " + std::to_string(matrix.cols) + "), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write("\x93NUMPY\x01\x00", 8);
    const uint16_t length = static_cast<uint16_t>(header.size());
    const char bytes[] = {static_cast<char>(length & 255), static_cast<char>(length >> 8)};
    file.write(bytes, 2);
    file.write(header.data(), header.size());
    for (int y = 0; y < matrix.rows; ++y)
        file.write(reinterpret_cast<const char *>(matrix.ptr<float>(y)),
                   matrix.cols * sizeof(float));
}

cv::Mat readNpy(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    char prefix[8];
    file.read(prefix, 8);
    if (std::string(prefix, 6) != std::string("\x93NUMPY", 6) || (prefix[6] != 1 && prefix[6] != 2))
        throw std::runtime_error("Unsupported NPY header");
    unsigned char bytes[4] = {};
    const int lengthBytes = prefix[6] == 1 ? 2 : 4;
    file.read(reinterpret_cast<char *>(bytes), lengthBytes);
    uint32_t length = 0;
    for (int i = 0; i < lengthBytes; ++i)
        length |= uint32_t(bytes[i]) << (8 * i);
    if (length == 0 || length > 65536)
        throw std::runtime_error("Invalid NPY header size");
    std::string header(length, '\0');
    file.read(header.data(), length);
    if (!std::regex_search(header, std::regex("['\"]descr['\"]\\s*:\\s*['\"]<f4['\"]")) ||
        !std::regex_search(header, std::regex("['\"]fortran_order['\"]\\s*:\\s*False")))
        throw std::runtime_error("NPY must contain C-order little-endian float32");
    std::smatch match;
    if (!std::regex_search(
            header, match,
            std::regex("['\"]shape['\"]\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,?\\s*\\)")))
        throw std::runtime_error("NPY must be a two-dimensional array");
    const auto rows = std::stoll(match[1]), cols = std::stoll(match[2]);
    if (rows <= 0 || cols <= 0 || rows > 100000 || cols > 100000 || rows * cols > 100000000)
        throw std::runtime_error("Invalid or oversized NPY dimensions");
    const auto payload = rows * cols * sizeof(float);
    const auto start = file.tellg();
    file.seekg(0, std::ios::end);
    if (file.tellg() - start != payload)
        throw std::runtime_error("NPY payload size does not match shape");
    file.seekg(start);
    cv::Mat result(static_cast<int>(rows), static_cast<int>(cols), CV_32F);
    file.read(reinterpret_cast<char *>(result.data), payload);
    return result;
}

float percentile(std::vector<float> values, double quantile) {
    std::sort(values.begin(), values.end());
    double index = quantile * (values.size() - 1);
    size_t low = static_cast<size_t>(index), high = std::min(low + 1, values.size() - 1);
    return static_cast<float>(values[low] + (values[high] - values[low]) * (index - low));
}

cv::Mat depthPreview(const cv::Mat &values, const std::vector<float> &valid) {
    cv::Mat gray(values.size(), CV_8U, cv::Scalar(0)), vis;
    if (!valid.empty()) {
        float low = percentile(valid, 0.02), high = percentile(valid, 0.98);
        if (high <= low)
            high = low + 1e-6f;
        for (int y = 0; y < values.rows; ++y)
            for (int x = 0; x < values.cols; ++x) {
                float value = values.at<float>(y, x);
                if (std::isfinite(value))
                    gray.at<uchar>(y, x) = static_cast<uchar>(
                        std::clamp((value - low) / (high - low), 0.0f, 1.0f) * 255);
            }
    }
    cv::applyColorMap(gray, vis, cv::COLORMAP_TURBO);
    for (int y = 0; y < values.rows; ++y)
        for (int x = 0; x < values.cols; ++x)
            if (!std::isfinite(values.at<float>(y, x)))
                vis.at<cv::Vec3b>(y, x) = {0, 0, 0};
    return vis;
}

void saveDepth(const fs::path &folder, const cv::Mat &depth) {
    if (depth.empty() || depth.type() != CV_32FC1)
        throw std::invalid_argument("Depth must be a nonempty float32 matrix");
    std::vector<float> valid;
    valid.reserve(depth.total());
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            const float value = depth.at<float>(y, x);
            if (std::isfinite(value) && value > 0)
                valid.push_back(value);
        }
    }
    fs::create_directories(folder);
    writeNpy(folder / "depth_from_disp_meter.npy", depth);
    writeImage(folder / "depth_from_disp_vis.png", depthPreview(depth, valid));
    fs::remove(folder / "depth_from_disp_vis_original.png");
    fs::remove(folder / "consistency_mask.png");
    std::cout << "Depth: " << depth.cols << 'x' << depth.rows << "; output: " << folder;
    if (!valid.empty())
        std::cout << "; median=" << percentile(valid, 0.5) << " m";
    std::cout << std::endl;
}

struct Viewer {
    cv::Mat depth, image;
    cv::Size original_size;
    const std::string title = "Depth Viewer - click a pixel, Esc to exit";
    static constexpr int header_height = 70;
    void show(const std::string &first, const std::string &second = "",
              cv::Point point = {-1, -1}) {
        cv::Mat frame(image.rows + header_height, image.cols, CV_8UC3, cv::Scalar(0));
        image.copyTo(frame(cv::Rect(0, header_height, image.cols, image.rows)));
        for (const auto &[line, baseline] : {std::pair{first, 28}, std::pair{second, 58}}) {
            if (line.empty())
                continue;
            const double scale = 0.55;
            cv::putText(frame, line, {12, baseline}, cv::FONT_HERSHEY_SIMPLEX, scale,
                        {255, 255, 255}, 1, cv::LINE_AA);
        }
        if (point.x >= 0)
            cv::drawMarker(frame, {point.x, point.y + header_height}, {255, 255, 255}, cv::MARKER_CROSS, 17,
                           2);
        cv::imshow(title, frame);
    }
    static void mouse(int event, int x, int y, int, void *context) {
        auto &self = *static_cast<Viewer *>(context);
        y -= header_height;
        if (event != cv::EVENT_LBUTTONDOWN || x < 0 || y < 0 || x >= self.depth.cols ||
            y >= self.depth.rows)
            return;
        float value = self.depth.at<float>(y, x);
        std::ostringstream coordinate, depth_text;
        coordinate << "Original pixel: x=" << x * self.original_size.width / self.depth.cols
                   << ", y=" << y * self.original_size.height / self.depth.rows;
        if (std::isfinite(value) && value > 0)
            depth_text << std::fixed << std::setprecision(4) << "Depth: " << value << " m ("
                       << std::setprecision(1) << value * 1000 << " mm)";
        else
            depth_text << "No valid depth";
        self.show(coordinate.str(), depth_text.str(), {x, y});
    }
};

void showDepth(const fs::path &folder, cv::Size original_size) {
    Viewer viewer;
    if (original_size.width <= 0 || original_size.height <= 0)
        throw std::invalid_argument("Original image size must be positive");
    viewer.original_size = original_size;
    viewer.depth = readNpy(folder / "depth_from_disp_meter.npy");
    viewer.image = cv::imread((folder / "depth_from_disp_vis.png").string());
    if (viewer.image.empty() || viewer.image.size() != viewer.depth.size())
        throw std::runtime_error("Preview/depth dimensions mismatch");
    cv::namedWindow(viewer.title, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(viewer.title, Viewer::mouse, &viewer);
    viewer.show("Click a pixel to show depth", "Esc / Q: exit");
    while (cv::getWindowProperty(viewer.title, cv::WND_PROP_VISIBLE) >= 1) {
        int key = cv::waitKey(30) & 255;
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }
    cv::destroyAllWindows();
}

} // namespace waft_stereo
