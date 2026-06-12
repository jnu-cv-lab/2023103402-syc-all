import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 读取测试图像 ----------------------
img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# 检查图片是否读取成功
if img1 is None or img2 is None:
    raise ValueError("图片读取失败，请检查 images/ 目录下是否存在 box.png 和 box_in_scene.png")

# ---------------------- 2. 创建 ORB 检测器（作业要求1、2） ----------------------
# 使用 cv2.ORB_create()，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# ---------------------- 3. 检测关键点和计算描述子（作业要求3） ----------------------
kp1, des1 = orb.detectAndCompute(img1, None)  # 第一幅图
kp2, des2 = orb.detectAndCompute(img2, None)  # 第二幅图

# ---------------------- 4. 可视化关键点（作业要求4） ----------------------
# 画关键点，用绿色圆圈标记
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# 保存可视化结果（用于提交）
cv2.imwrite("results/box_kp.png", img1_kp)
cv2.imwrite("results/box_in_scene_kp.png", img2_kp)

# ---------------------- 5. 输出关键点数量（作业要求5） ----------------------
print("===== 关键点检测结果 =====")
print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")

# ---------------------- 6. 输出描述子维度（作业要求6） ----------------------
print("\n===== 描述子信息 =====")
print(f"box.png 描述子维度: {des1.shape if des1 is not None else '无描述子'}")
print(f"box_in_scene.png 描述子维度: {des2.shape if des2 is not None else '无描述子'}")

# ---------------------- 可选：显示对比图（方便你直接查看） ----------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img1_kp, cmap='gray')
axs[0].set_title("box.png ORB Keypoints")
axs[1].imshow(img2_kp, cmap='gray')
axs[1].set_title("box_in_scene.png ORB Keypoints")
plt.tight_layout()
plt.savefig("results/keypoints_comparison.png")
plt.close()