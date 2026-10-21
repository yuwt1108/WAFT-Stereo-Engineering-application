# WAFT-Stereo C++ 工程

处理流程：原始左右图 → 双目校正 → WAFT 视差推理 → 米制深度 → 保存并点击查看。

## 在 VS 中运行

1. 打开 `build/WAFTStereo.sln`；提示工程有外部修改时选择重新加载。
2. 选择 **Release / x64**，将 `waft_stereo` 设为启动项目，调试命令参数留空。
3. 按 **Ctrl+F5**，程序从 `main.cpp` 执行，保存结果并打开深度窗口。

主入口不再解析命令参数或提供分阶段命令。VS 构建通过 `Directory.Build.targets` 自动处理重复的 Path 环境项，调试环境已配置动态库路径。

## 输入与结果

当前读取 `data/input/1021/1021_50cm/left.png` 和 `right.png`，来自本次1224×1024采集的示例。更换路径时，修改 `main.cpp` 的 `input_folder`；按left、right查找PNG/JPG/JPEG，读写位置不依赖工作目录。

输入必须是同一时刻、当前相机标定对应的 **1224×1024 原始未校正左右图**，格式为 BGR、`CV_8UC3`。接入相机时，用获取的左右 `cv::Mat` 替换两次 `imread`：

```cpp
#include "stereo_frame_processor.hpp"

cv::Mat values = waft_stereo::process_frames(left, right);
```

`process_frames(left, right)` 返回 `CV_32F`、612×512 米制深度图；`main()` 将其按原图宽高比调整为 612×512 后放入 `values`，保存并显示。无效值为 `NaN`。像素 `(x, y)` 的深度使用 `values.at<float>(y, x)`，使用前判断数值是否有限且大于零。

帧接口使用 `alpha=0.2`，将 1224×1024 校正图缩放为 612×512，只对左右图进行一次模型推理。若需更换推理尺寸，修改 `main.cpp` 的 `inference_size`；输出高度根据原图宽高比计算，深度计算按实际水平缩放比例调整焦距。输出缩放使用最近邻插值，不会增加真实深度细节。相机标定在 `stereo.cpp` 的 `makeRectification()` 中，来自 `20261021_水下标定测试/matlab.mat`。K、D、R、T已成套替换；参数说明见 `stereo_calibration_opencv_1224_1024.txt`。旧文件 `stereo_calibration_opencv_1920_1200.txt` 仅供历史追溯。当前校正接口会拒绝不匹配尺寸的原图，不会自动将旧图缩放后套新标定。

## 输出

每次运行按左图所在文件夹的名称保存到 `data/output/<输入文件夹名>`，例如输入 `data/input/1021/1021_50cm/left.png` 和 `right.png` 时，输出目录为 `data/output/1021_50cm`。不同名称的输入文件夹分别保存；输入文件夹名称相同时覆盖该输出目录中的同名结果。输出基础目录可在 `main.cpp` 中修改。传入 `process_frames(left, right, output, inference_size)` 时保存：

- `left_rect.png`、`right_rect.png`：1224×1024 左右校正图，无损保存。
- `rectified_preview.png`：左右校正图并排显示，绿色水平线辅助检查对齐。
- `relative_disparity.png`：612×512 彩色相对视差图，按有效正视差的最小值/最大值归一化，无效像素为黑色。颜色越接近暖色，视差越大，通常物体越近；该图片不能用于计算米制深度。
- `uncertainty.png`：模型辅助通道的可视化。

深度保存到对应输出目录中的 `depth` 子目录：

- `depth_from_disp_meter.npy`：612×512 的二维 float32 米制深度数据。
- `depth_from_disp_vis.png`：612×512、使用 2%～98% 深度分位着色的预览图。

鼠标点击窗口可查询本次运行的深度，按 Esc/Q 关闭。点击坐标按横、纵各自的比例换算为原始 1224×1024 图上的坐标；深度值取自对应的 612×512 输出像素。窗口从 NPY 读取数值，PNG 只用于显示。当前不进行左右一致性校验，因此非重叠或遮挡区域的深度也可能显示，但可靠性较低。推理或写入失败时，本次目录可能只有部分结果。

## 代码职责

| 文件 | 职责 |
| --- | --- |
| `src/main.cpp` | 读取左右图、调用函数、接收深度、保存和显示 |
| `src/stereo_frame_processor.hpp/.cpp` | 接收两个 cv::Mat，组织完整深度处理流程 |
| `src/stereo.hpp/.cpp` | 校正、模型加载与预处理、推理、视差转深度 |
| `src/depth_result_handler.hpp/.cpp` | NPY/图片读写、结果保存和点击窗口 |
| `CMakeLists.txt` | VS2019/C++17 构建和调试配置 |
| `Directory.Build.targets` | VS 构建前处理重复的 Path 环境项 |

## 环境与模型

VS2019/v142、x64 Release、C++17、OpenCV 4.12、LibTorch 2.5/CUDA 12.1，需要 CUDA GPU。依赖路径在 `CMakeLists.txt` 中设置。

程序当前加载 `data/model/waft_B.pt`。在 `main()` 中修改 `model_path` 即可切换模型，例如改成 `data/model/waft_L.pt`，然后重新编译。调用为 `process_frames(left, right, output, inference_size, model_path)`；省略模型路径时默认使用 B 版。同一路径连续处理帧时复用模型。LibTorch 原生加载，运行时不调用 Python。`tools/export_model.py` 用于更换权重时导出，需要父目录的 Python 模型源码与 .pth 权重。

两参数调用 `process_frames(left, right)` 只返回深度；主程序使用带输出目录的调用，同时保存中间预览。旧的分阶段命令、历史验证目录和 PowerShell 启动脚本已清理。`build` 保留当前 VS 解决方案和构建所需文件。

当前深度计算筛选正的有限视差，尚未过滤校正黑边和非重叠区域；数值核对不代表实物测距精度验证。

## 本次标定与数据版本

当前使用1224×1024原始图，默认612×512推理与输出，保持原图比例；`main.cpp`中的`inference_size`和`model_path`仍可修改。深度使用校正后P1的焦距、P2计算的基线（约0.0597246363443 m），再按实际水平缩放比调整焦距，不直接用原始K替代校正K。

相机的采集模式、左右顺序应与新标定一致；仅尺寸相同并不足以证明标定适用。旧输入与历史输出目录不作重算，当前默认输入已指向1021的新示例。报告中的减2.5 cm起点补偿未写入本程序。

## GitHub发布版本的数据与权重

本仓库不附带data实际数据，只保留目录结构；克隆后请按[data/README.md](data/README.md)放置原始左右图，并按[data/model/README.md](data/model/README.md)准备模型。官方权重入口为[WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo#weights-on-huggingface)及[Hugging Face SynLarge](https://huggingface.co/MemorySlices/WAFT-Stereo/tree/main/SynLarge)。官方`.pth`需转换为当前C++接口兼容的TorchScript `.pt`，不能仅更名。

本仓库不包含build目录。请从仓库根目录执行`cmake -S C++ -B C++/build -G "Visual Studio 16 2019" -A x64`生成VS解决方案，按本机路径设置依赖后再构建。
