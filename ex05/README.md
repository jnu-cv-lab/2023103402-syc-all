# 计算机视觉实验五：几何变换与透视校正

## 实验环境
- 操作系统：Ubuntu / WSL2  
- 开发工具：VS Code  
- 编程语言：Python 3.12  
- 依赖库：OpenCV、NumPy、Matplotlib  

## 实验目的

1. 理解并实现三种基础几何变换：
   - 相似变换（Similarity Transform）
   - 仿射变换（Affine Transform）
   - 透视变换（Perspective Transform）

2. 分析不同变换对几何性质的影响：
   - 直线是否保持为直线
   - 平行关系是否保持
   - 垂直关系是否保持
   - 圆是否仍为圆

3. 掌握透视畸变校正方法：
   - 基于四点透视变换恢复平面图像
   - 理解角点选取对结果的影响

## 项目结构

```
├── images/
│   └── paper.jpg
├── test.png
├── transform_comparison.png
├── paper_corrected.jpg
└── main.py
```

## 实验原理与实现步骤

### 1. 测试图像构建

使用 OpenCV 构造包含：
- 矩形
- 圆
- 平行线
- 垂直线  


### 2. 相似变换

```python
cv2.getRotationMatrix2D(center, angle, scale)
```

特点：
- 保持角度
- 保持平行与垂直
- 圆仍为圆

### 3. 仿射变换

```python
cv2.getAffineTransform(pts1, pts2)
```

特点：
- 保持直线和平行
- 不保持角度
- 圆变椭圆

### 4. 透视变换

```python
cv2.getPerspectiveTransform(pts1, pts2)
```

特点：
- 只保持直线
- 平行关系消失
- 存在透视效果


### 5. 透视校正

```python
M = cv2.getPerspectiveTransform(pts1, pts2)
img_corrected = cv2.warpPerspective(img, M, (w, h))
```

## 实验过程问题
- 图像歪斜，并且只显示一小部分疑似内容被裁剪。可能是角点选取不准或者点顺序错误。后来我不断调整四个角点保证顺序左上 → 右上 → 左下 → 右下逐步逼近真实纸张边界，成功恢复完整纸张基本无畸变，透视效果明显消除

## 实验结果
- 相似变换无结构变化，仿射变换：形变明显，透视变换：远近效果明显，校正后图像基本恢复正常  


## 实验结论

1. 相似变换保持了所有几何性质，仿射变换破坏角度但保持平行，透视变换最灵活但破坏最多 

2. 校正的关键：精度最重要，顺序必须正确

3 要反复调参，一定要保证顺序左上 → 右上 → 左下 → 右下

## 运行方式

```bash
python main.py
```

---

## 学生信息
作者：孙勇超
学号：2023103402