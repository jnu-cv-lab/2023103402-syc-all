import cv2
import numpy as np

# ---------------------- 1. 读取图像（如果任务1已读取可跳过） ----------------------
img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# ---------------------- 2. 重新检测ORB特征点和描述子（任务1部分） ----------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# ---------------------- 3. 创建BFMatcher暴力匹配器（作业要求1、2、3） ----------------------
# 使用cv2.NORM_HAMMING，开启crossCheck=True
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ---------------------- 4. 执行匹配 ----------------------
matches = bf.match(des1, des2)

# ---------------------- 5. 按匹配距离从小到大排序（作业要求4） ----------------------
matches = sorted(matches, key=lambda x: x.distance)

# ---------------------- 6. 可视化前30个匹配结果（作业要求5） ----------------------
# 画出前30个匹配
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:30], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 保存匹配结果图
cv2.imwrite("results/orb_matches_top30.png", img_matches)

# ---------------------- 7. 输出总匹配数量（作业要求6） ----------------------
print("===== ORB 特征匹配结果 =====")
print(f"总匹配数量: {len(matches)}")
print(f"已保存前30个匹配的可视化结果到 results/orb_matches_top30.png")