import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

#读取图像
img = cv2.imread("_20231211070702.jpg")
if img is None:
    print(" 图片读取失败")
    exit()

# 输出图像基本信息
h, w = img.shape[:2]
channels = img.shape[2] if len(img.shape) == 3 else 1
dtype = img.dtype
print("===== 图像基本信息 =====")
print(f"宽度 : {w}")
print(f"高度 : {h}")
print(f"通道数 : {channels}")
print(f"数据类型 : {dtype}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Original Image")
plt.savefig("original_image.png")
plt.close()
print(" 原图已保存为 original_image.png")

# 转换为灰度图 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title("Grayscale Image")
plt.savefig("grayscale_image.png")
plt.close()
print(" 灰度图已保存为 grayscale_image.png")

# 保存灰度图
cv2.imwrite("grayscale_result.jpg", img_gray)
print(" 灰度图结果已保存为 grayscale_result.jpg")

# NumPy 操作：输出像素值 + 裁剪保存
pixel_val = img[100, 100]
print(f"\n 像素(100,100)的值: {pixel_val}")

crop_region = img[0:200, 0:200]
cv2.imwrite("cropped_region.jpg", crop_region)
print(" 裁剪区域已保存为 cropped_region.jpg")


