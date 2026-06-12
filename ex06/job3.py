import cv2
import numpy as np

# ---------------------- 1. 读取图像 + ORB检测（如果前面已做可跳过） ----------------------
img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# ---------------------- 2. BFMatcher匹配（任务2部分，这里重新实现，保证独立） ----------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# ---------------------- 3. 提取对应点坐标（作业要求1） ----------------------
# 从匹配结果中提取两张图的点坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ---------------------- 4. 使用cv2.findHomography + RANSAC（作业要求2、3、4） ----------------------
# 方法：cv2.RANSAC，重投影误差阈值设置为5.0
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# 把mask转成一维数组，方便筛选
matchesMask = mask.ravel().tolist()

# ---------------------- 5. 筛选RANSAC内点并可视化（作业要求5） ----------------------
# 只保留RANSAC筛选后的内点匹配
draw_params = dict(matchColor=(0, 255, 0),  # 匹配线颜色（绿色）
                   singlePointColor=None,
                   matchesMask=matchesMask,  # 只画内点
                   flags=2)

# 画出RANSAC后的匹配结果
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

# 保存结果图
cv2.imwrite("results/orb_matches_ransac.png", img_ransac)

# ---------------------- 6. 输出统计信息（作业要求6） ----------------------
total_matches = len(matches)
num_inliers = sum(matchesMask)  # mask中1的数量就是内点数量
inlier_ratio = num_inliers / total_matches

print("===== RANSAC 剔除错误匹配结果 =====")
print(f"总匹配数量: {total_matches}")
print(f"RANSAC 内点数量: {num_inliers}")
print(f"内点比例: {inlier_ratio:.4f}")
print("\nHomography 矩阵:")
print(H)

print("\n✅ 已保存 RANSAC 后的匹配图到 results/orb_matches_ransac.png")