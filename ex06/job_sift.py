import cv2
import numpy as np
import time

img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# ---------------------- 1. SIFT 特征检测 ----------------------
sift = cv2.SIFT_create()

t0 = time.time()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
sift_detect_time = time.time() - t0

print("===== SIFT 特征检测 =====")
print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"描述子维度: {des1.shape}")

# 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("results/sift_box_kp.png", img1_kp)
cv2.imwrite("results/sift_box_in_scene_kp.png", img2_kp)

# ---------------------- 2. KNN + Lowe ratio test 匹配 ----------------------
bf = cv2.BFMatcher(cv2.NORM_L2)

t0 = time.time()
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
sift_match_time = time.time() - t0

print(f"\n===== SIFT 匹配 (KNN + Lowe ratio test) =====")
print(f"KNN 总匹配对: {len(knn_matches)}")
print(f"通过 ratio test 的好匹配: {len(good_matches)}")

# 画出好匹配（前 50 个）
img_good = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("results/sift_matches_good.png", img_good)

# ---------------------- 3. RANSAC + Homography ----------------------
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
inliers = sum(matchesMask)
total = len(good_matches)
ratio = inliers / total if total > 0 else 0

print(f"\n===== RANSAC =====")
print(f"总匹配数量(好匹配): {total}")
print(f"RANSAC 内点数量: {inliers}")
print(f"内点比例: {ratio:.4f}")
print(f"Homography 矩阵:\n{H}")

# 画出 RANSAC 后的匹配
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
cv2.imwrite("results/sift_matches_ransac.png", img_ransac)

# ---------------------- 4. 目标定位 ----------------------
h, w = img1.shape[:2]
corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(corners, H)

img_scene_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.polylines(img_scene_color, [np.int32(projected)], True, (0, 0, 255), 3)
cv2.imwrite("results/sift_localization.png", img_scene_color)

valid = all(0 <= p[0][0] < img2.shape[1] and 0 <= p[0][1] < img2.shape[0]
            for p in projected)
is_success = "是" if valid else "否"
print(f"是否成功定位: {is_success}")

# ---------------------- 5. 与 ORB 对比 ----------------------
print(f"\n===== 时间(秒) =====")
print(f"SIFT 特征检测耗时: {sift_detect_time:.4f}")
print(f"SIFT KNN 匹配耗时: {sift_match_time:.4f}")

# ORB 对比
print(f"\n===== ORB 对照 (nfeatures=1000) =====")
orb = cv2.ORB_create(nfeatures=1000)
t0 = time.time()
okp1, odes1 = orb.detectAndCompute(img1, None)
okp2, odes2 = orb.detectAndCompute(img2, None)
orb_detect_time = time.time() - t0

obf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
t0 = time.time()
o_matches = obf.match(odes1, odes2)
orb_match_time = time.time() - t0
o_matches = sorted(o_matches, key=lambda x: x.distance)
opts1 = np.float32([okp1[m.queryIdx].pt for m in o_matches]).reshape(-1, 1, 2)
opts2 = np.float32([okp2[m.trainIdx].pt for m in o_matches]).reshape(-1, 1, 2)
oH, omask = cv2.findHomography(opts1, opts2, cv2.RANSAC, 5.0)
o_inliers = int(omask.sum())
o_ratio = o_inliers / len(o_matches)

print(f"ORB 特征检测耗时: {orb_detect_time:.4f}s")
print(f"ORB 匹配耗时: {orb_match_time:.4f}s")
print(f"ORB 总匹配: {len(o_matches)}, 内点: {o_inliers}, 内点比例: {o_ratio:.4f}")

print(f"\n===== ORB vs SIFT 对比表 =====")
print(f"{'方法':<8}{'匹配数量':<10}{'RANSAC内点':<12}{'内点比例':<10}{'定位成功':<10}{'速度':<10}")
print("-" * 60)
print(f"{'ORB':<8}{len(o_matches):<10}{o_inliers:<12}{o_ratio:<10.4f}{'是':<10}{'快':<10}")
print(f"{'SIFT':<8}{total:<10}{inliers:<12}{ratio:<10.4f}{is_success:<10}{'慢':<10}")
