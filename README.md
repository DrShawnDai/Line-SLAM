# Line-SLAM
### 1. Introduction

空间直线的参数化表示方法、三角化重建、以及重投影误差优化。

Author：咸菜爱嗑盐（Billibilli）
视频链接：
https://www.bilibili.com/video/BV1WnqqYyEzB/?vd_source=ab405eb800b446f6d678bf5b3c73f36a

### 2. Dependencies

We have tested the libarary in **MacOS 13.6** and **Ubuntu 20.04**, but it should be easy to compile in other platforms. Other Prerequisites as below:

- c++14 Compiler
- Pangolin 0.8
- OpenCV 4.5.0
- Eigen 3.4.0
- Ceres 2.2.0

### 3. Build & Run

cd examples

./run_plucker.sh （对应视频第一节：参数化表示方法及三角化重建）

./run_optimize.sh（对应视频第二节：重投影误差优化）

### 4. TODO List

- [x] line feature extraction & matching based on "LSD + LBD"
- [ ] line feature extraction & matching based on "GlueStick"
