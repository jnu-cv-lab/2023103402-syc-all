import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import math

# -------------------------- 1. 图像读取与预处理 --------------------------
def read_image(path):
    """读取图像并转为灰度图"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("图片读取失败，请检查路径！")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# 读取原始图像
original_img = read_image("images/original.jpg")
height, width = original_img.shape
print(f"原始图像尺寸：{height}×{width}")

# -------------------------- 2. 图像下采样 --------------------------
def downsample_direct(img, scale=0.5):
    """直接下采样（缩小1/2）"""
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return downsampled

def downsample_smooth(img, scale=0.5):
    """先高斯平滑再下采样"""
    # 高斯平滑（核大小5×5，标准差1.5）
    smoothed = cv2.GaussianBlur(img, (5, 5), 1.5)
    # 下采样
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    downsampled = cv2.resize(smoothed, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return downsampled

# 执行下采样
down_direct = downsample_direct(original_img)
down_smooth = downsample_smooth(original_img)

# 保存下采样图像
cv2.imwrite("images/downsampled_direct.jpg", down_direct)
cv2.imwrite("images/downsampled_smooth.jpg", down_smooth)
print("下采样图像已保存")

# -------------------------- 3. 图像恢复（上采样） --------------------------
def restore_image(img, original_size, method):
    """
    恢复图像到原始尺寸
    method: nearest / bilinear / bicubic
    """
    methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }
    restored = cv2.resize(img, (original_size[1], original_size[0]), interpolation=methods[method])
    return restored

# 基于“平滑后下采样”的图像恢复（也可对比直接下采样的恢复效果）
restore_nearest = restore_image(down_smooth, original_img.shape, "nearest")
restore_bilinear = restore_image(down_smooth, original_img.shape, "bilinear")
restore_bicubic = restore_image(down_smooth, original_img.shape, "bicubic")

# 保存恢复图像
cv2.imwrite("images/restored_nearest.jpg", restore_nearest)
cv2.imwrite("images/restored_bilinear.jpg", restore_bilinear)
cv2.imwrite("images/restored_bicubic.jpg", restore_bicubic)
print("恢复图像已保存")

# -------------------------- 4. 空间域评价（MSE、PSNR） --------------------------
def calculate_mse(img1, img2):
    """计算均方误差MSE"""
    # 确保两张图像尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calculate_psnr(img1, img2):
    """计算峰值信噪比PSNR"""
    mse = calculate_mse(img1, img2)
    if mse == 0:  # 图像完全相同
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# 计算各恢复图像的MSE和PSNR
metrics = {
    "最近邻内插": {
        "MSE": calculate_mse(original_img, restore_nearest),
        "PSNR": calculate_psnr(original_img, restore_nearest)
    },
    "双线性内插": {
        "MSE": calculate_mse(original_img, restore_bilinear),
        "PSNR": calculate_psnr(original_img, restore_bilinear)
    },
    "双三次内插": {
        "MSE": calculate_mse(original_img, restore_bicubic),
        "PSNR": calculate_psnr(original_img, restore_bicubic)
    }
}

# 打印空间域评价结果
print("\n=== 空间域评价结果 ===")
for method, values in metrics.items():
    print(f"{method} - MSE: {values['MSE']:.2f}, PSNR: {values['PSNR']:.2f} dB")

# 绘制空间域对比图
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title("原始图像")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(restore_nearest, cmap='gray')
plt.title(f"最近邻恢复\nMSE: {metrics['最近邻内插']['MSE']:.2f}, PSNR: {metrics['最近邻内插']['PSNR']:.2f} dB")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(restore_bilinear, cmap='gray')
plt.title(f"双线性恢复\nMSE: {metrics['双线性内插']['MSE']:.2f}, PSNR: {metrics['双线性内插']['PSNR']:.2f} dB")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(restore_bicubic, cmap='gray')
plt.title(f"双三次恢复\nMSE: {metrics['双三次内插']['MSE']:.2f}, PSNR: {metrics['双三次内插']['PSNR']:.2f} dB")
plt.axis('off')

plt.tight_layout()
plt.savefig("results/spatial_comparison.png", dpi=300)
plt.close()
print("空间域对比图已保存到results目录")

# -------------------------- 5. 傅里叶变换（FFT）分析 --------------------------
def fft_analysis(img):
    """对图像进行傅里叶变换，返回频谱（中心化）"""
    # 傅里叶变换
    f = np.fft.fft2(img)
    # 中心化
    f_shift = np.fft.fftshift(f)
    # 计算幅度谱（对数缩放，便于显示）
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum

# 计算各图像的频谱
fft_original = fft_analysis(original_img)
fft_down = fft_analysis(down_smooth)
fft_nearest = fft_analysis(restore_nearest)
fft_bicubic = fft_analysis(restore_bicubic)

# 绘制频谱图
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(fft_original, cmap='gray')
plt.title("原始图像频谱")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(fft_down, cmap='gray')
plt.title("下采样图像频谱")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(fft_nearest, cmap='gray')
plt.title("最近邻恢复频谱")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(fft_bicubic, cmap='gray')
plt.title("双三次恢复频谱")
plt.axis('off')

plt.tight_layout()
plt.savefig("results/fft_spectrums.png", dpi=300)
plt.close()
print("傅里叶频谱图已保存到results目录")

# -------------------------- 6. DCT变换分析 --------------------------
def dct_analysis(img):
    """对图像进行DCT变换，返回DCT系数（归一化）和低频能量占比"""
    # 对图像分块（8×8），也可直接对整图做DCT
    h, w = img.shape
    # 归一化图像到0-1
    img_normalized = img / 255.0
    # DCT变换
    dct_coeffs = dct(dct(img_normalized.T, norm='ortho').T, norm='ortho')
    
    # 计算低频能量占比（取前10%的低频系数）
    total_energy = np.sum(np.square(dct_coeffs))
    # 中心化DCT系数（便于显示）
    dct_shift = np.fft.fftshift(dct_coeffs)
    # 取低频区域（中心区域）
    low_freq_h = int(h * 0.1)
    low_freq_w = int(w * 0.1)
    center_h, center_w = h // 2, w // 2
    low_freq_region = dct_shift[
        center_h - low_freq_h : center_h + low_freq_h,
        center_w - low_freq_w : center_w + low_freq_w
    ]
    low_energy = np.sum(np.square(low_freq_region))
    low_energy_ratio = (low_energy / total_energy) * 100
    
    # 归一化DCT系数便于显示
    dct_display = (dct_coeffs - np.min(dct_coeffs)) / (np.max(dct_coeffs) - np.min(dct_coeffs))
    return dct_display, low_energy_ratio

# 计算各图像的DCT系数和低频能量占比
dct_original, ratio_original = dct_analysis(original_img)
dct_nearest, ratio_nearest = dct_analysis(restore_nearest)
dct_bilinear, ratio_bilinear = dct_analysis(restore_bilinear)
dct_bicubic, ratio_bicubic = dct_analysis(restore_bicubic)

# 打印DCT低频能量占比
print("\n=== DCT低频能量占比 ===")
print(f"原始图像: {ratio_original:.2f}%")
print(f"最近邻恢复: {ratio_nearest:.2f}%")
print(f"双线性恢复: {ratio_bilinear:.2f}%")
print(f"双三次恢复: {ratio_bicubic:.2f}%")

# 绘制DCT系数图
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(dct_original, cmap='gray')
plt.title(f"原始图像DCT系数\n低频占比: {ratio_original:.2f}%")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(dct_nearest, cmap='gray')
plt.title(f"最近邻恢复DCT系数\n低频占比: {ratio_nearest:.2f}%")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dct_bilinear, cmap='gray')
plt.title(f"双线性恢复DCT系数\n低频占比: {ratio_bilinear:.2f}%")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(dct_bicubic, cmap='gray')
plt.title(f"双三次恢复DCT系数\n低频占比: {ratio_bicubic:.2f}%")
plt.axis('off')

plt.tight_layout()
plt.savefig("results/dct_coefficients.png", dpi=300)
plt.close()
print("DCT系数图已保存到results目录")

print("\n=== 实验全部完成！所有结果已保存到对应目录 ===")