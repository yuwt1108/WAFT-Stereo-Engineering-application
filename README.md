# WAFT-Stereo Engineering Application

WAFT-Stereo的Windows C++工程应用：原始双目图像 → 双目校正 → 视差推理 → 米制深度 → 保存与点击查询。

工程在 [`C++`](C++/) 中，详细用法见 [`C++/README.md`](C++/README.md)。使用LibTorch原生推理，正常运行时不调用Python。

## 当前配置

- 原始图像：1224×1024，对应2026-10-21水下重新标定。
- 模型输入：612×512，保持原图比例；在`C++/src/main.cpp`中修改。
- 模型：`waft_B.pt`；一次左视差推理，不做左右一致性校验。
- 校正：`alpha=0.2`；K、D、R、T已匹配新采集模式。
- 环境：Windows、Visual Studio 2019/v142、C++17、OpenCV 4.12、LibTorch 2.5 CUDA 12.1。

## 数据与模型放置

`C++/data`只保留目录结构、`.gitkeep`和说明，不包含图片、模型或历史测试输出。现有历史目录是目录模板，不表示仓库附带这些数据。

| 内容 | 默认位置 |
|---|---|
| 左原始图 | `C++/data/input/1021/1021_50cm/left.png` |
| 右原始图 | `C++/data/input/1021/1021_50cm/right.png` |
| C++推理模型 | `C++/data/model/waft_B.pt` |
| 本次运行结果 | `C++/data/output/1021_50cm/` |

原始图片支持PNG/JPG/JPEG，必须与标定的相机采集模式和左右顺序一致。修改`main.cpp`中的`input_folder`、`model_path`、`inference_size`即可更换输入和模型。完整目录说明见 [`C++/data/README.md`](C++/data/README.md)。

## 官方权重来源

- 原始项目：[princeton-vl/WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo)。
- 官方README权重入口：[Weights on Hugging Face](https://github.com/princeton-vl/WAFT-Stereo#weights-on-huggingface)。
- 官方权重目录：[MemorySlices/WAFT-Stereo/SynLarge](https://huggingface.co/MemorySlices/WAFT-Stereo/tree/main/SynLarge)。
- B版：[DAv2B-4.pth下载](https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/SynLarge/DAv2B-4.pth?download=true)。
- L版：[DAv2L-5.pth下载](https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/SynLarge/DAv2L-5.pth?download=true)。

官方发布的是PyTorch `.pth`检查点。当前C++加载的是经过本工程导出工具处理的TorchScript `.pt`，**不能直接把`.pth`改名为`.pt`**。转换说明见 [`C++/data/model/README.md`](C++/data/model/README.md)。本仓库不附带权重。

## 构建和启动

先安装所需依赖，并按本机位置修改`C++/CMakeLists.txt`中的OpenCV、LibTorch和CUDA路径。然后在仓库根目录生成VS工程：

```powershell
cmake -S C++ -B C++/build -G "Visual Studio 16 2019" -A x64
cmake --build C++/build --config Release --target waft_stereo -- /m:1
```

打开`C++/build/WAFTStereo.sln`，选择Release/x64、将`waft_stereo`设为启动项目，再运行。先准备好上述图片和TorchScript模型。VS调试配置包含依赖DLL搜索路径。

## 标定与校正检查

- [`新标定参数`](C++/stereo_calibration_opencv_1224_1024.txt)
- [`新标定alpha对比`](C++/rectification_review_1224_1024/all_alpha_comparison.jpg)
- [`新标定校正检查说明`](C++/rectification_review_1224_1024/REPORT.md)
- `stereo_calibration_opencv_1920_1200.txt`和同尺寸校正检查目录为历史记录，不供当前默认输入使用。

校正对比图为已生成的分析结果。原分析脚本及Python模型源码位于原完整工程中，本仓库只发布C++目录。

保留了C++工程原有的[`LICENSE`](C++/LICENSE)；算法及第三方模型的来源和使用条件请同时参阅官方仓库。
