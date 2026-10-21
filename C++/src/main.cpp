#include "depth_result_handler.hpp"
#include "stereo_frame_processor.hpp"

#include <filesystem>
#include <iostream>
#include <initializer_list>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace waft_stereo;

// 工程路径由构建配置提供，VS 的当前工作目录不影响默认读写位置。
const fs::path ROOT = WAFT_PROJECT_DIR;   //CMakeLists.txt里面配置

int main() {
    try {
        // 修改输入文件夹即可；同名图片优先使用 PNG，其次 JPG、JPEG。
        const fs::path input_folder = ROOT / "data/input/1021/1021_50cm";

        const auto find_image = [&input_folder](const std::string &name) -> fs::path {
            for (const char *extension : {".png", ".jpg", ".jpeg"}) {
                const fs::path path = input_folder / (name + extension);
                if (fs::is_regular_file(path))
                    return path;
            }
            throw std::runtime_error("Cannot find " + name +
                                     " (.png/.jpg/.jpeg) in " + input_folder.string());
        };
        const fs::path left_path = find_image("left");
        const fs::path right_path = find_image("right");

        // 输入须为新标定对应的1224×1024原始未校正左右图，BGR、同一时刻拍摄。
        // 接入相机时，将下面两次 imread 替换成相机获取的 cv::Mat 左右帧。
        cv::Mat left = cv::imread(left_path.string(), cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(right_path.string(), cv::IMREAD_COLOR);

        if (left.empty() || right.empty())
            throw std::runtime_error("Cannot read left/right image: " +
                                     left_path.string() + " / " + right_path.string());

        // 使用左图所在文件夹的名称；同名输入目录再次运行时覆盖同名结果。
        const fs::path output = ROOT / "data/output" / left_path.parent_path().filename();

        // 在这里修改模型输入尺寸；612×512为原图的一半，保持宽高比。
        const cv::Size inference_size(612, 512);

        // 在这里修改模型文件名，例如 waft_B.pt 或 waft_L.pt    自己选择。
        const fs::path model_path = ROOT / "data/model/waft_B.pt";

        cv::Mat inferred_depth = process_frames(left, right, output, inference_size, model_path);

        // 输出保持原图宽高比；最近邻缩放保留米制深度和无效值，不生成新细节。
        const cv::Size output_size(inference_size.width,
                                   inference_size.width * left.rows / left.cols);
        cv::Mat values;
        cv::resize(inferred_depth, values, output_size, 0, 0, cv::INTER_NEAREST);
        // 获取像素 (x, y) 的深度：values.at<float>(y, x)，注意先判断是否为有限值。

        // 校正图、视差图与深度结果统一保存到该输入目录对应的输出文件夹。
        saveDepth(output / "depth", values);

        // 鼠标点击查询深度，按 Esc 或 Q 关闭窗口。
        showDepth(output / "depth", left.size());
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
