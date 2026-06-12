# 计算机视觉实验六：基于 OpenCV 的局部特征检测、描述与图像匹配

## 实验环境
- 操作系统：Ubuntu / WSL2
- 开发工具：VS Code
- 编程语言：Python 3
- 依赖库：OpenCV (含 contrib，用于 SIFT) + NumPy + Matplotlib

## 实验目的
1. 掌握 ORB 局部特征检测与二进制描述子的原理与使用
2. 掌握基于 BFMatcher + Hamming 距离的特征匹配，以及 crossCheck 的作用
3. 理解 RANSAC 在剔除错误匹配中的几何一致性思想
4. 利用 Homography 完成模板图到场景图的目标定位
5. 通过参数对比实验，分析 `nfeatures` 对匹配数量与内点比例的影响
6. 选做：完成 SIFT + KNN + Lowe ratio test 的对比实验

## 项目结构
```
syc-experiment06/
├── images/
│   ├── box.png                         # 模板图
│   └── box_in_scene.png                # 场景图
├── results/
│   ├── box_kp.png                      # 任务1：模板图 ORB 关键点
│   ├── box_in_scene_kp.png             # 任务1：场景图 ORB 关键点
│   ├── keypoints_comparison.png        # 任务1：两图关键点并排对比
│   ├── orb_matches_top30.png           # 任务2：前 30 个 ORB 匹配
│   ├── orb_matches_ransac.png          # 任务3：RANSAC 后的内点匹配
│   ├── target_localization.png         # 任务4：场景图中的目标定位
│   ├── localization_nfeatures_500.png  # 任务6：nfeatures=500 定位结果
│   ├── localization_nfeatures_1000.png # 任务6：nfeatures=1000 定位结果
│   ├── localization_nfeatures_2000.png # 任务6：nfeatures=2000 定位结果
│   ├── sift_box_kp.png                 # 选做：SIFT 模板图关键点
│   ├── sift_box_in_scene_kp.png        # 选做：SIFT 场景图关键点
│   ├── sift_matches_good.png           # 选做：SIFT 经 ratio test 后的好匹配
│   ├── sift_matches_ransac.png         # 选做：SIFT RANSAC 内点匹配
│   └── sift_localization.png           # 选做：SIFT 目标定位结果
├── job1.py                             # 任务1：ORB 关键点与描述子
├── job2.py                             # 任务2：ORB 暴力匹配
├── job3.py                             # 任务3：RANSAC 剔除错误匹配
├── job4.py                             # 任务4：Homography 目标定位
├── job6.py                             # 任务6：nfeatures 参数对比
└── job_sift.py                         # 选做：SIFT vs ORB 对比
```

## 运行方式
```bash
# 安装依赖（SIFT 需要 contrib 包，OpenCV ≥4.4 已合并到主包）
pip install opencv-python opencv-contrib-python numpy matplotlib

# 依次运行
python job1.py
python job2.py
python job3.py
python job4.py
python job6.py
python job_sift.py
```

---

## 实验结果

### 任务 1：ORB 关键点检测（`nfeatures=1000`）

| 图像 | 关键点数量 | 描述子维度 |
|------|------------|------------|
| box.png | 865 | (865, 32) |
| box_in_scene.png | 1000 | (1000, 32) |

ORB 描述子为 32 字节 = **256 位二进制** 描述子。

### 任务 2：ORB 暴力匹配（BFMatcher + NORM_HAMMING + crossCheck）
- 总匹配数量：**287**
- 已按距离升序排序，前 30 个匹配可视化保存于 `results/orb_matches_top30.png`

### 任务 3：RANSAC 剔除错误匹配（重投影阈值 5.0）
- 总匹配数量：287
- RANSAC 内点数量：**52**
- 内点比例：**0.1812**
- Homography 矩阵：
```
[[ 4.344e-01  -1.271e-01   1.180e+02]
 [-1.734e-02   4.630e-01   1.601e+02]
 [-3.368e-04  -1.274e-04   1.000e+00]]
```

### 任务 4：目标定位
将 box.png 的四个角点经 Homography 投影至场景图，使用 `cv2.polylines` 绘出红色四边形边框。**定位成功**：四边形位置与场景中真实物体高度吻合。

### 任务 6：参数对比实验

| nfeatures | 模板图关键点 | 场景图关键点 | 匹配数量 | RANSAC 内点 | 内点比例 | 是否成功定位 |
|-----------|--------------|--------------|----------|-------------|----------|--------------|
| 500       | 453          | 500          | 149      | 32          | 0.2148   | 是           |
| 1000      | 865          | 1000         | 287      | 52          | 0.1812   | 是           |
| 2000      | 1589         | 1999         | 511      | 66          | 0.1292   | 是           |

**分析：**
- `nfeatures` 越大，模板图与场景图的关键点数量及总匹配数量都近似线性增长。
- 内点数量随 `nfeatures` 增长而增加（32→52→66），但增速远慢于匹配数量。
- **内点比例反而下降**（0.2148→0.1812→0.1292）。原因是新增的关键点多落在响应较弱、纹理重复或边界模糊的区域，产生了大量"看起来相似但几何不一致"的错误匹配。
- 三组参数都成功定位，说明只要内点的绝对数量足以稳定估计 Homography（≥4 即可，通常 ≥10 已较稳健），定位就能成功。
- **结论：特征点不是越多越好。** 数量过少时内点不足、几何估计不稳；数量过多时引入大量噪声匹配，反而压低内点比例并增加 RANSAC 的计算量。`nfeatures=1000` 是较为均衡的选择。

### 选做任务：SIFT 对比实验

SIFT 检测：box.png 关键点 604，box_in_scene.png 关键点 969，描述子维度 (N, 128)。
匹配采用 KNN(k=2) + Lowe ratio test (阈值 0.75)。

| 方法 | 匹配数量 | RANSAC 内点 | 内点比例 | 是否成功定位 | 运行速度（主观） |
|------|----------|-------------|----------|--------------|------------------|
| ORB  | 287      | 52          | 0.1812   | 是           | 快（二进制描述子 + Hamming 距离）|
| SIFT | 80       | 75          | **0.9375** | 是         | 较慢（浮点 128 维 + L2）|

**对比分析：**
- SIFT 经 Lowe ratio test 后保留的匹配数虽较少（80 < 287），但**几乎全部是正确匹配**，内点比例高达 0.94。
- ORB 不做 ratio test 而采用 crossCheck，对错误匹配的过滤较弱，内点比例低（0.18），但绝对内点数量足以支撑 RANSAC。
- 两种方法在本实验中都成功完成了目标定位，**SIFT 的匹配质量更高、Homography 估计更稳定**；ORB 速度上有优势，适合实时场景。


## 学生信息
作者：孙勇超  
学号：2023103402
