import glob
import os
import cv2
import numpy as np

# PAD/打印棋盘格参数：9 x 6 内角点，方格边长 25 mm。
CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0
IMAGE_GLOB = "images/*.*"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints = []
os.makedirs("detected_corners", exist_ok=True)

images = sorted(glob.glob(IMAGE_GLOB))
if len(images) < 15:
    raise SystemExit("需要至少 15 张标定图片，请把实拍图放入 images 文件夹。")

image_size = None
success_count = 0
for filename in images:
    img = cv2.imread(filename)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ok:
        print("未检测到角点:", filename)
        continue
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    success_count += 1
    preview = img.copy()
    cv2.drawChessboardCorners(preview, CHECKERBOARD, corners2, ok)
    cv2.imwrite(os.path.join("detected_corners", os.path.basename(filename)), preview)
    print("检测成功:", filename)

if success_count < 15:
    raise SystemExit(f"有效标定图片不足 15 张，当前成功 {success_count} 张。")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

total_error = 0
for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    total_error += error
mean_error = total_error / len(objpoints)

print("\n===== 标定结果 =====")
print("有效图片数量:", success_count)
print("RMS 重投影误差:", ret)
print("平均重投影误差(pixel):", mean_error)
print("相机内参矩阵 K:")
print(K)
print("畸变参数 [k1, k2, p1, p2, k3]:")
print(dist.ravel())

sample = cv2.imread(images[0])
h, w = sample.shape[:2]
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(sample, K, dist, None, newK)
cv2.imwrite("undistorted_example.jpg", undistorted)
