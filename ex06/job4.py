import cv2
import numpy as np

# ---------------------- 任务1：ORB特征检测 ----------------------
img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
cv2.imwrite("results/box_kp.png", img1_kp)
cv2.imwrite("results/box_in_scene_kp.png", img2_kp)

print("===== 关键点检测结果 =====")
print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"box.png 描述子维度: {des1.shape}")
print(f"box_in_scene.png 描述子维度: {des2.shape}")

# ---------------------- 任务2：ORB特征匹配 ----------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("results/orb_matches_top30.png", img_matches)

print("\n===== ORB 特征匹配结果 =====")
print(f"总匹配数量: {len(matches)}")

# ---------------------- 任务3：RANSAC剔除错误匹配 ----------------------
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
cv2.imwrite("results/orb_matches_ransac.png", img_ransac)

total_matches = len(matches)
num_inliers = sum(matchesMask)
inlier_ratio = num_inliers / total_matches

print("\n===== RANSAC 剔除错误匹配结果 =====")
print(f"总匹配数量: {total_matches}")
print(f"RANSAC 内点数量: {num_inliers}")
print(f"内点比例: {inlier_ratio:.4f}")
print("\nHomography 矩阵:")
print(H)

# ---------------------- 任务4：目标定位 ----------------------
h, w = img1.shape[:2]
corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
projected_corners = cv2.perspectiveTransform(corners, H)

img_scene = img2.copy()
cv2.polylines(img_scene, [np.int32(projected_corners)], True, (0, 0, 255), 3)
cv2.imwrite("results/target_localization.png", img_scene)

print("\n===== 目标定位结果 =====")
print("✅ 已将box.png的四个角点投影到场景图中，并画出红色边框")
print("定位结果已保存为: results/target_localization.png")