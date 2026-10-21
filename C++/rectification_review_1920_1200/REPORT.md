# 双目校正检查

检查对象：`stereo_range_waft.py`、`matlab.mat`、`stereo_befor/Left_preview.jpg` 和 `Right_preview.jpg`。

## 结论

现有代码使用已标定的针孔相机双目校正：`stereoRectify` → `initUndistortRectifyMap` → `remap`，流程正确。明显裁剪放大主要来自固定 `alpha=0` 的视野策略，不是另一次显式裁剪或 resize。

从 MATLAB 文件的对象存储数据中核查到左右标定 ImageSize 都为 `[1200, 1920]`（高、宽），与当前原图相符。MATLAB 内参的主点换成 OpenCV 坐标后，和脚本 K1/K2 一致；畸变参数、相对位姿也与导出的标定文本及脚本一致。主点明显偏离画面中心，但不是本次代码复制或图像尺寸顺序导致的错误；是否为可靠物理标定，仍需标定图片和测距实测确认。

## 当前图片的视野对比

| alpha | 校正焦距 fx | 相对原始 fx 的比例 | 左/右输出像素有合法源坐标的比例 |
|---|---:|---:|---:|
| 0 | 2613.847 | 1.633 | 100.0% / 100.0% |
| 0.5 | 2095.105 | 1.309 | 97.7% / 99.5% |
| 1 | 1576.363 | 0.985 | 79.6% / 80.5% |

比例是焦距比，不能认为畸变图上所有位置均匀放大相同倍数。合法源坐标比例统计 remap 的坐标是否处于图像内，不代表双目共有视场或可靠深度比例。

`alpha_comparison.jpg` 从上到下为原图、alpha=0、0.5、1；每行左右排列同一组相机图像。

## 对齐检查

原图提取 SIFT，使用最近邻比值 0.7 筛选出 243 组匹配，再用基础矩阵 RANSAC（1.5 像素）保留 167 组。将匹配点用当前标定和校正投影矩阵变换后，alpha=0 时纵向误差的中位数为 0.472 像素，90 分位为 1.409 像素。

这是当前场景的一次特征匹配检查，存在误匹配与筛选偏差，不能代替全视场标定板验证。不同 alpha 会改变输出焦距和像素尺度，不应直接比较纵向像素误差来断言哪个 alpha 的几何精度更高。

## 建议

1. 保持已标定的双目校正流程，增加 alpha 选择；当前场景可先评估 0.5，重视最大视野时评估 1。
2. 有黑边时保存有效区域/掩码。深度计算还需检查右图对应位置 `x-d` 是否有效，避免给黑边或非重叠区显示虚假深度。
3. 保存 P1、P2、Q、R1、R2、图像尺寸和 alpha，与图片及 K.txt 同批输出。
4. 左右图若裁剪，应使用一致的公共裁剪区域，并从两侧主点减去裁剪偏移；不能裁剪后再拉伸回旧尺寸却沿用旧 K。
5. 用覆盖全视场的标定板和已知距离目标验证。水下测量应使用当前镜头、防水舱和实际介质条件下的标定；固定针孔模型对折射的近似需实测工作距离范围。

## 工业及官方参考

- OpenCV 已标定针孔双目校正及 alpha 定义：https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html
- HALCON 校正映射及校正后的相机参数：https://www.mvtec.com/doc/halcon/2411/en/gen_binocular_rectification_map.html
- Basler 校正、校准及验证：https://docs.baslerweb.com/camera-calibration
- Basler 使用校正后焦距计算深度：https://docs.baslerweb.com/stereovisard/rc_visard/en/stereo_matching
- OpenCV 鱼眼相机校正：https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html
- 水下折射模型及针孔近似的范围：https://arxiv.org/abs/2403.08640

本次仅检查并生成对比图；未更改校正脚本或覆盖 stereo_after 中的输出。
