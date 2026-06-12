import cv2
import numpy as np

# ---------------------- 配置 ----------------------
img1 = cv2.imread("images/box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# 要测试的三组nfeatures参数
nfeatures_list = [500, 1000, 2000]
results = []

for nfeatures in nfeatures_list:
    print(f"\n===== 测试 nfeatures = {nfeatures} =====")

    # 1. ORB特征检测
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    template_kp_count = len(kp1)
    scene_kp_count = len(kp2)

    # 2. ORB特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_count = len(matches)

    # 3. RANSAC剔除错误匹配
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    inlier_count = sum(matchesMask)
    inlier_ratio = inlier_count / match_count if match_count > 0 else 0

    # 4. 目标定位（判断是否成功）
    is_success = "否"
    if H is not None:
        h, w = img1.shape[:2]
        corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(corners, H)
        # 检查投影点是否在场景图范围内
        valid = all(0 <= p[0][0] < img2.shape[1] and 0 <= p[0][1] < img2.shape[0] for p in projected_corners)
        if valid:
            # 保存结果图
            img_scene = img2.copy()
            cv2.polylines(img_scene, [np.int32(projected_corners)], True, (0, 0, 255), 3)
            cv2.imwrite(f"results/localization_nfeatures_{nfeatures}.png", img_scene)
            is_success = "是"

    # 记录结果
    results.append({
        "nfeatures": nfeatures,
        "template_kp": template_kp_count,
        "scene_kp": scene_kp_count,
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "is_success": is_success
    })

    print(f"模板图关键点数量: {template_kp_count}")
    print(f"场景图关键点数量: {scene_kp_count}")
    print(f"匹配数量: {match_count}")
    print(f"RANSAC内点数: {inlier_count}")
    print(f"内点比例: {inlier_ratio:.4f}")
    print(f"是否成功定位: {is_success}")

# ---------------------- 打印对比表格 ----------------------
print("\n===== 参数对比实验结果汇总 =====")
print(f"{'nfeatures':<10} {'模板图关键点':<12} {'场景图关键点':<12} {'匹配数量':<8} {'内点数':<8} {'内点比例':<8} {'定位成功':<6}")
print("-" * 70)
for res in results:
    print(f"{res['nfeatures']:<10} {res['template_kp']:<12} {res['scene_kp']:<12} {res['match_count']:<8} {res['inlier_count']:<8} {res['inlier_ratio']:<8} {res['is_success']:<6}")