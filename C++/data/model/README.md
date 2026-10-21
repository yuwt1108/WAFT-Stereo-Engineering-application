# 模型来源与放置

## 官方检查点

模型来自[princeton-vl/WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo)，其README的[权重说明](https://github.com/princeton-vl/WAFT-Stereo#weights-on-huggingface)指向[Hugging Face官方目录](https://huggingface.co/MemorySlices/WAFT-Stereo/tree/main/SynLarge)。

| 模型 | 官方PyTorch检查点 | 本工程C++模型文件名 |
|---|---|---|
| B | [DAv2B-4.pth](https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/SynLarge/DAv2B-4.pth?download=true) | `waft_B.pt`（默认） |
| L | [DAv2L-5.pth](https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/SynLarge/DAv2L-5.pth?download=true) | `waft_L.pt` |

官方`.pth`需导出为TorchScript `.pt`，不能仅修改文件扩展名。导出的模型需带`contract.txt=normalized_rgb_symmetric_pad16_dynamic_v1`元数据，输入预处理和双输出格式必须与`src/stereo.cpp`匹配。

## 导出依赖

`tools/export_model.py`是原完整工程使用的导出工具，依赖它的同级Python模型工程：

```text
完整工程/
├── C++/tools/export_model.py
└── Python/WATF_demo_project/
    ├── deploy/waft_deploy.py
    ├── algorithms/
    ├── model/
    ├── thirdparty/
    ├── bridgedepth/
    └── configs/SynLarge/
```

本仓库只上传C++，不包含上述Python模型源码；单独下载本仓库和`.pth`还不足以运行导出工具。需要先补齐原完整工程的Python依赖，或者使用已通过本工具导出的兼容`.pt`。官方Python仓库本身不提供本工程的`deploy/waft_deploy.py`适配文件。

补齐Python模型工程并安装对应依赖后，在仓库根目录使用B版配置导出，例如：

```powershell
python C++/tools/export_model.py --checkpoint C++/data/model/DAv2B-4.pth --config Python/WATF_demo_project/configs/SynLarge/DAv2B-4.yaml --left C++/data/input/1021/1021_50cm/left.png --right C++/data/input/1021/1021_50cm/right.png --width 612 --height 512 --output C++/data/model/waft_B.pt
```

L版需同时使用`DAv2L-5.pth`、匹配的`DAv2L-5.yaml`及`waft_L.pt`输出名。修改`main.cpp`的`model_path`即可切换导出的模型。C++推理时不需要Python。
