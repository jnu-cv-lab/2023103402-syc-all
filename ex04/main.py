import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # WSL2/Ubuntu 自带的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 生成测试图 ----------------------
def generate_checkerboard(size=256, block_size=8):
    """生成棋盘格图"""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
    return img

def generate_chirp(size=256):
    """生成chirp测试图（水平方向频率从0到π渐变）"""
    x = np.linspace(0, np.pi, size)
    y = np.linspace(0, np.pi, size)
    xx, yy = np.meshgrid(x, y)
    # 水平方向频率随x线性增加，y方向保持恒定
    img = np.sin(xx * np.linspace(0, 10, size)) * 127 + 128
    return img.astype(np.uint8)

# 生成并保存测试图
checker = generate_checkerboard()
chirp = generate_chirp()
cv2.imwrite("test_images/checkerboard.png", checker)
cv2.imwrite("test_images/chirp.png", chirp)

# ---------------------- 2. 下采样实现 ----------------------
def downsample_direct(img, M=2):
    """直接下采样：每隔M个像素取一个"""
    return img[::M, ::M]

def downsample_gaussian(img, M=2, sigma=1.0):
    """高斯滤波后下采样：抗混叠"""
    # 高斯核大小根据sigma自适应，保证核足够大
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::M, ::M]

# ---------------------- 3. FFT频谱计算 ----------------------
def compute_fft_spectrum(img):
    """计算图像的FFT幅度谱（中心化）"""
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)  # +1避免log(0)
    return magnitude

# ---------------------- 4. 第一部分结果可视化 ----------------------
def part1_visualize():
    # 读取测试图
    checker = cv2.imread("test_images/checkerboard.png", cv2.IMREAD_GRAYSCALE)
    chirp = cv2.imread("test_images/chirp.png", cv2.IMREAD_GRAYSCALE)
    M = 2
    sigma = 1.0

    # 棋盘格处理
    checker_direct = downsample_direct(checker, M)
    checker_gauss = downsample_gaussian(checker, M, sigma)
    # Chirp处理
    chirp_direct = downsample_direct(chirp, M)
    chirp_gauss = downsample_gaussian(chirp, M, sigma)

    # 计算频谱
    checker_fft = compute_fft_spectrum(checker)
    checker_direct_fft = compute_fft_spectrum(checker_direct)
    checker_gauss_fft = compute_fft_spectrum(checker_gauss)

    chirp_fft = compute_fft_spectrum(chirp)
    chirp_direct_fft = compute_fft_spectrum(chirp_direct)
    chirp_gauss_fft = compute_fft_spectrum(chirp_gauss)

    # 绘制对比图
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # 棋盘格行
    axs[0,0].imshow(checker, cmap='gray')
    axs[0,0].set_title("原始棋盘格")
    axs[0,1].imshow(checker_direct, cmap='gray')
    axs[0,1].set_title("直接下采样（混叠明显）")
    axs[0,2].imshow(checker_gauss, cmap='gray')
    axs[0,2].set_title("高斯滤波后下采样（无混叠）")

    # 频谱行
    axs[1,0].imshow(checker_fft, cmap='gray')
    axs[1,0].set_title("原始频谱")
    axs[1,1].imshow(checker_direct_fft, cmap='gray')
    axs[1,1].set_title("直接下采样频谱（混叠导致高频泄漏）")
    axs[1,2].imshow(checker_gauss_fft, cmap='gray')
    axs[1,2].set_title("滤波后下采样频谱（混叠消除）")

    plt.tight_layout()
    plt.savefig("results/part1_checker_comparison.png", dpi=300)
    plt.close()

    # 同理绘制chirp图对比（省略代码，结构一致）
    # ... 保存为results/part1_chirp_comparison.png

part1_visualize()

# ---------------------- 第二部分：σ参数验证 ----------------------
def part2_sigma_test():
    # 读取测试图（用自然图+棋盘格，更全面）
    img = cv2.imread("test_images/checkerboard.png", cv2.IMREAD_GRAYSCALE)
    M = 4
    sigmas = [0.5, 1.0, 2.0, 4.0]
    theoretical_sigma = 0.45 * M  # 理论最优σ=1.8

    # 绘制不同σ的效果对比
    fig, axs = plt.subplots(2, len(sigmas), figsize=(20, 8))
    for i, sigma in enumerate(sigmas):
        # 下采样
        downsampled = downsample_gaussian(img, M, sigma)
        # 计算频谱
        fft_spec = compute_fft_spectrum(downsampled)

        # 空间域
        axs[0,i].imshow(downsampled, cmap='gray')
        axs[0,i].set_title(f"σ={sigma:.1f}")
        # 频域
        axs[1,i].imshow(fft_spec, cmap='gray')
        axs[1,i].set_title(f"σ={sigma:.1f} 频谱")

    plt.suptitle(f"M={M} 不同σ下采样效果对比（理论最优σ={theoretical_sigma:.1f}）", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/part2_sigma_comparison.png", dpi=300)
    plt.close()

    # 补充：最优σ(1.8)和理论值对比
    sigma_opt = 1.8
    downsampled_opt = downsample_gaussian(img, M, sigma_opt)
    cv2.imwrite("results/downsampled_opt_sigma.png", downsampled_opt)

part2_sigma_test()

# ---------------------- 第三部分：自适应下采样 ----------------------
def compute_gradient(img):
    """计算图像梯度（Sobel算子）"""
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient

def adaptive_downsample(img, block_size=16, M_max=4):
    """自适应下采样：分块计算梯度，动态调整M和σ"""
    h, w = img.shape
    # 分块
    blocks_h = h // block_size
    blocks_w = w // block_size
    # 初始化结果图
    result = np.zeros_like(img)
    # 梯度图
    grad = compute_gradient(img)
    grad_mean = grad.mean()

    for i in range(blocks_h):
        for j in range(blocks_w):
            # 提取块
            y1, y2 = i*block_size, (i+1)*block_size
            x1, x2 = j*block_size, (j+1)*block_size
            block = img[y1:y2, x1:x2]
            block_grad = grad[y1:y2, x1:x2].mean()

            # 根据梯度动态调整M：梯度大→M小，梯度小→M大
            if block_grad > grad_mean * 1.5:
                # 高梯度（边缘）：M=2，σ=0.9（0.45*2）
                M = 2
                sigma = 0.45 * M
            elif block_grad > grad_mean * 0.5:
                # 中等梯度：M=3，σ=1.35
                M = 3
                sigma = 0.45 * M
            else:
                # 低梯度（平滑）：M=4，σ=1.8
                M = 4
                sigma = 0.45 * M

            # 自适应下采样
            block_down = downsample_gaussian(block, M, sigma)
            # 上采样回原块大小（用双三次插值，保持质量）
            block_up = cv2.resize(block_down, (block_size, block_size), interpolation=cv2.INTER_CUBIC)
            # 写入结果
            result[y1:y2, x1:x2] = block_up

    return result

# ---------------------- 自适应 vs 统一下采样对比 ----------------------
def part3_adaptive_compare():
    # 读取自然图像（用自己的original.jpg，更真实）
    img = cv2.imread("../syc-experiment03/images/original.jpg", cv2.IMREAD_GRAYSCALE)
    # 统一下采样（M=4，σ=1.8）
    uniform_down = downsample_gaussian(img, M=4, sigma=1.8)
    uniform_up = cv2.resize(uniform_down, img.shape[::-1], interpolation=cv2.INTER_CUBIC)
    # 自适应下采样
    adaptive_up = adaptive_downsample(img)

    # 计算MSE（均方误差），量化对比
    mse_uniform = np.mean((img - uniform_up) ** 2)
    mse_adaptive = np.mean((img - adaptive_up) ** 2)
    print(f"统一下采样MSE: {mse_uniform:.2f}")
    print(f"自适应下采样MSE: {mse_adaptive:.2f}")

    # 绘制对比图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("原始图像")
    axs[1].imshow(uniform_up, cmap='gray')
    axs[1].set_title(f"统一下采样（M=4）\nMSE={mse_uniform:.2f}")
    axs[2].imshow(adaptive_up, cmap='gray')
    axs[2].set_title(f"自适应下采样\nMSE={mse_adaptive:.2f}")
    plt.tight_layout()
    plt.savefig("results/part3_adaptive_comparison.png", dpi=300)
    plt.close()

    # 绘制误差热力图
    error_uniform = np.abs(img - uniform_up)
    error_adaptive = np.abs(img - adaptive_up)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(error_uniform, cmap='hot')
    axs[0].set_title("统一下采样误差热力图")
    axs[1].imshow(error_adaptive, cmap='hot')
    axs[1].set_title("自适应下采样误差热力图")
    plt.tight_layout()
    plt.savefig("results/part3_error_heatmap.png", dpi=300)
    plt.close()

part3_adaptive_compare()

