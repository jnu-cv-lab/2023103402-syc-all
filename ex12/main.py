import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Task 1: Sinusoidal Position Encoding
# ============================================================

def sinusoidal_pe(max_len, d_model):
    PE = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
    PE[:, 0::2] = torch.sin(pos / div)
    PE[:, 1::2] = torch.cos(pos / div)
    return PE

max_len = 50
d_model = 64
PE = sinusoidal_pe(max_len, d_model)
print(f"Sinusoidal PE shape: {PE.shape}")

plt.figure(figsize=(10, 4))
plt.imshow(PE.numpy(), aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Dimension index')
plt.ylabel('Position index')
plt.title('Sinusoidal Position Encoding')
plt.tight_layout()
plt.savefig('sinusoidal_pe.png', dpi=100)
plt.close()
print("saved: sinusoidal_pe.png")

# ============================================================
# Task 2: 二维向量旋转
# ============================================================

def rotate_2d(x, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ x

angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
v0 = np.array([1.0, 0.0])

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
for t in angles:
    rv = rotate_2d(v0, t)
    ax.annotate('', xy=rv, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
    ax.text(rv[0] * 1.2, rv[1] * 1.2, f'{np.degrees(t):.0f}°', fontsize=8)
ax.set_title('2D Vector Rotation')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('rotate_2d.png', dpi=100)
plt.close()
print("saved: rotate_2d.png")

# ============================================================
# Task 3: 高维 RoPE
# ============================================================

def rope_single(x, pos):
    """对单个向量 x (d,) 施加位置 pos 的 RoPE"""
    d = x.shape[-1]
    assert d % 2 == 0
    half = d // 2
    theta = torch.pow(torch.tensor(10000.0), -torch.arange(0, half).float() / half)
    angles = pos * theta
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_a - x2 * sin_a,
                      x1 * sin_a + x2 * cos_a], dim=-1)

def rope_sequence(x):
    """对序列 x (seq_len, d) 逐位置施加 RoPE"""
    return torch.stack([rope_single(x[m], m) for m in range(x.shape[0])])

seq_len = 10
d = 16
x_test = torch.randn(seq_len, d)
x_rope = rope_sequence(x_test)
print(f"RoPE output shape: {x_rope.shape}")

# ============================================================
# Task 4: 对比 E+pos 和 RoPE 的输入方式
# ============================================================

print("\n--- E+pos vs RoPE ---")

E = torch.randn(seq_len, d)
PE_small = sinusoidal_pe(seq_len, d)
X_epos = E + PE_small
print(f"E+pos: word embedding + PE, shape = {X_epos.shape}")

Q = torch.randn(seq_len, d)
K = torch.randn(seq_len, d)
Q_rope = rope_sequence(Q)
K_rope = rope_sequence(K)
print(f"RoPE:  Q_rope shape = {Q_rope.shape}, K_rope shape = {K_rope.shape}")

# ============================================================
# Task 5: 数值实验验证 RoPE 的相对位置性质
# ============================================================

print("\n--- 验证 RoPE 相对位置性质：dot(R(m)q, R(n)k) 只依赖 m-n ---")

q = torch.randn(d)
k = torch.randn(d)
delta = 5

pairs = [(0, 5), (3, 8), (10, 15), (20, 25), (50, 55)]
scores = []
for m, n in pairs:
    score = torch.dot(rope_single(q, m), rope_single(k, n)).item()
    scores.append(score)
    print(f"  m={m:2d}, n={n:2d}, delta={n-m}, dot={score:.6f}")

print(f"\n最大差值: {max(scores) - min(scores):.2e}  (趋近于0，验证成功)")

# 可视化：不同相对位移下的平均 attention score
deltas = list(range(-30, 31))
avg_scores = []
for dlt in deltas:
    trial_scores = []
    for _ in range(300):
        qr = torch.randn(d)
        kr = torch.randn(d)
        m = 40
        n = m + dlt
        if n < 0:
            m = abs(dlt)
            n = 0
        trial_scores.append(torch.dot(rope_single(qr, m), rope_single(kr, n)).item())
    avg_scores.append(np.mean(trial_scores))

plt.figure(figsize=(8, 4))
plt.plot(deltas, avg_scores, marker='o', markersize=3)
plt.xlabel('Relative position (m - n)')
plt.ylabel('Average dot product')
plt.title('RoPE: Average dot product vs relative position')
plt.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
plt.savefig('rope_relative.png', dpi=100)
plt.close()
print("saved: rope_relative.png")

# 对比图：固定相对距离 delta=5，改变绝对位置，E+pos 不稳定，RoPE 稳定
print("\n--- 对比 E+pos 与 RoPE：固定 delta=5，改变绝对位置 ---")
positions = list(range(0, 50, 2))
epos_scores = []
rope_scores_cmp = []
PE_full = sinusoidal_pe(100, d)
q_fixed = torch.randn(d)
k_fixed = torch.randn(d)

for p in positions:
    q_ep = q_fixed + PE_full[p]
    k_ep = k_fixed + PE_full[p + 5]
    epos_scores.append(torch.dot(q_ep, k_ep).item())

    rq = rope_single(q_fixed, p)
    rk = rope_single(k_fixed, p + 5)
    rope_scores_cmp.append(torch.dot(rq, rk).item())

plt.figure(figsize=(9, 4))
plt.plot(positions, epos_scores, marker='s', markersize=4, label='E+pos (delta=5)')
plt.plot(positions, rope_scores_cmp, marker='o', markersize=4, label='RoPE (delta=5)')
plt.xlabel('Absolute position m')
plt.ylabel('Attention score (q·k)')
plt.title('Attention score vs absolute position (fixed relative delta=5)')
plt.legend()
plt.tight_layout()
plt.savefig('compare_epos_rope.png', dpi=100)
plt.close()
print("saved: compare_epos_rope.png")

# ============================================================
# Task 6: 为什么 RoPE 比 E+pos 更巧妙
# ============================================================

print("\n--- RoPE vs E+pos 原理对比 ---")
print("""
E+pos 方式:
  输入 = E(token) + PE(pos)
  做 attention 时: score = (E+PE)·W_q · ((E+PE)·W_k)^T
  展开后包含 E·Wq·(PE·Wk)^T 等交叉项，内容与位置信息耦合

RoPE 方式:
  输入嵌入不变，只在计算 Q/K 时旋转
  q_m = R(m) · W_q · E(m),  k_n = R(n) · W_k · E(n)
  score = q_m^T · k_n = (R(m)q)^T (R(n)k)
        = q^T R(m)^T R(n) k = q^T R(n-m) k
  结果只含相对位置 (n-m)，内容与位置解耦

优势:
  1. 相对位置天然编码在点积中
  2. 词嵌入本身不受污染
  3. 外推性更好（绝对位置不影响相对关系）
""")

print("All done.")
