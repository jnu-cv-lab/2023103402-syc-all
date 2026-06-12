import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_image(size=500):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # 1. 画矩形
    cv2.rectangle(img, (100, 100), (200, 200), (0,0,0), 2)
    # 2. 画圆
    cv2.circle(img, (350, 250), 50, (0,0,0), 2)
    # 3. 画平行线（水平）
    cv2.line(img, (50, 300), (450, 300), (0,0,0), 2)
    cv2.line(img, (50, 350), (450, 350), (0,0,0), 2)
    # 4. 画垂直线
    cv2.line(img, (150, 50), (150, 450), (0,0,0), 2)
    cv2.line(img, (200, 50), (200, 450), (0,0,0), 2)

    cv2.imwrite("test.png", img)
    return img

img = create_test_image()


# 相似变换矩阵：旋转30度，缩放0.8倍
rows, cols = img.shape[:2]
angle = 30
scale = 0.8
M_similar = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
img_similar = cv2.warpAffine(img, M_similar, (cols, rows))

#仿射变换
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (cols, rows))

#透视变换
pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
pts2 = np.float32([[50,50], [cols-50,100], [50,rows-50], [cols-50,rows-100]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
img_perspective = cv2.warpPerspective(img, M_perspective, (cols, rows))

# 画对比图
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].imshow(img)
axs[0,0].set_title("Original")
axs[0,1].imshow(img_similar)
axs[0,1].set_title("Similarity")
axs[1,0].imshow(img_affine)
axs[1,0].set_title("Affine")
axs[1,1].imshow(img_perspective)
axs[1,1].set_title("Perspective")
plt.tight_layout()
plt.savefig("transform_comparison.png")

img = cv2.imread("images/paper.jpg")
h, w = img.shape[:2]

# 校准的 4 个角点（顺序：左上、右上、左下、右下）
pts1 = np.float32([
    [240, 340],   # 左上
    [730, 320],   # 右上
    [200, 860],   # 左下
    [780, 880]    # 右下
])

pts2 = np.float32([
    [0, 0],
    [w, 0],
    [0, h],
    [w, h]
])

# 透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
img_corrected = cv2.warpPerspective(img, M, (w, h))

cv2.imwrite("paper_corrected.jpg", img_corrected)