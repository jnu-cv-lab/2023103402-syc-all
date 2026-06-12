import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================================
# 环境检查
# ============================================================
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============================================================
# 数据加载（MNIST）
# ============================================================
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test       = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

train_size = 50000
val_size   = len(mnist_train_full) - train_size
mnist_train, mnist_val = random_split(mnist_train_full, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(mnist_val,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(mnist_test,  batch_size=64, shuffle=False, num_workers=2)

print(f"\nMNIST 训练集: {len(mnist_train)}, 验证集: {len(mnist_val)}, 测试集: {len(mnist_test)}")

# ============================================================
# 模型定义（复用实验08的 SimpleCNN）
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

# ============================================================
# 通用训练 / 评估函数
# ============================================================
criterion = nn.CrossEntropyLoss()

def train_model(model, optimizer, num_epochs=10, desc=''):
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(num_epochs):
        # ----- Train -----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            _, pred    = out.max(1)
            t_correct += pred.eq(labels).sum().item()
            t_total   += labels.size(0)

        tl = t_loss / len(train_loader)
        ta = t_correct / t_total
        train_losses.append(tl)
        train_accs.append(ta)

        # ----- Validation -----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item()
                _, pred    = out.max(1)
                v_correct += pred.eq(labels).sum().item()
                v_total   += labels.size(0)

        vl = v_loss / len(val_loader)
        va = v_correct / v_total
        val_losses.append(vl)
        val_accs.append(va)

        print(f"[{desc}] Epoch [{epoch+1}/{num_epochs}] "
              f"train loss: {tl:.4f}, train acc: {ta:.4f} | "
              f"val loss: {vl:.4f}, val acc: {va:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model):
    model.eval()
    all_preds, all_labels, all_imgs = [], [], []
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            _, pred     = out.max(1)
            correct    += pred.eq(labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_imgs.extend(imgs.cpu())

    return total_loss / len(test_loader), correct / total, all_preds, all_labels, all_imgs


# ============================================================
# Task 1：复用上次 CNN 模型，重新训练并记录训练过程
# ============================================================
print("\n" + "="*60)
print("Task 1: Retrain SimpleCNN (Adam, lr=0.001, 10 epochs)")
print("="*60)

model     = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
t1_train_losses, t1_val_losses, t1_train_accs, t1_val_accs = \
    train_model(model, optimizer, num_epochs=10, desc='Task1')

t1_test_loss, t1_test_acc, t1_all_preds, t1_all_labels, t1_all_imgs = evaluate_model(model)
print(f"\nTask1 Test Loss: {t1_test_loss:.4f}, Test Acc: {t1_test_acc:.4f}")

# 训练曲线
epochs_range = range(1, 11)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs_range, t1_train_losses, 'b-o', label='train loss')
ax1.plot(epochs_range, t1_val_losses,   'r-o', label='val loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Task 1 – Loss Curve (Adam lr=0.001)'); ax1.legend()

ax2.plot(epochs_range, t1_train_accs, 'b-o', label='train acc')
ax2.plot(epochs_range, t1_val_accs,   'r-o', label='val acc')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Task 1 – Accuracy Curve (Adam lr=0.001)'); ax2.legend()

plt.tight_layout()
plt.savefig('task1_train_curve.png', dpi=100)
plt.close()
print("saved: task1_train_curve.png")

# ============================================================
# Task 2：优化器对比（SGD / SGD+Momentum / Adam）
# ============================================================
print("\n" + "="*60)
print("Task 2: Optimizer Comparison")
print("="*60)

optimizer_configs = [
    ('SGD',          lambda m: optim.SGD(m.parameters(), lr=0.01)),
    ('SGD+Momentum', lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9)),
    ('Adam',         lambda m: optim.Adam(m.parameters(), lr=0.001)),
]

opt_results = {}
for opt_name, opt_fn in optimizer_configs:
    print(f"\n--- {opt_name} ---")
    m   = SimpleCNN().to(device)
    opt = opt_fn(m)
    tr_l, vl_l, tr_a, vl_a = train_model(m, opt, num_epochs=10, desc=opt_name)
    _, test_acc, _, _, _    = evaluate_model(m)
    opt_results[opt_name]   = dict(train_losses=tr_l, val_losses=vl_l,
                                   train_accs=tr_a,   val_accs=vl_a,
                                   test_acc=test_acc)
    print(f"{opt_name} test accuracy: {test_acc:.4f}")

# 4 子图对比
colors = {'SGD': 'steelblue', 'SGD+Momentum': 'darkorange', 'Adam': 'seagreen'}
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
titles = ['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy']
keys   = ['train_losses', 'val_losses', 'train_accs', 'val_accs']
ylabels= ['Loss', 'Loss', 'Accuracy', 'Accuracy']

for ax, title, key, ylabel in zip(axes.flat, titles, keys, ylabels):
    for name, res in opt_results.items():
        ax.plot(epochs_range, res[key], '-o', label=name, color=colors[name], markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_title(title);    ax.legend()

plt.suptitle('Task 2 – Optimizer Comparison (10 epochs, MNIST)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('task2_optimizer_comparison.png', dpi=100)
plt.close()
print("saved: task2_optimizer_comparison.png")

# 汇总表
print("\nTask 2 Summary:")
for name, res in opt_results.items():
    print(f"  {name:15s}: train_acc={res['train_accs'][-1]:.4f}, "
          f"val_acc={res['val_accs'][-1]:.4f}, test_acc={res['test_acc']:.4f}")

# ============================================================
# Task 3：学习率对比（Adam: 0.1 / 0.01 / 0.001）
# ============================================================
print("\n" + "="*60)
print("Task 3: Learning Rate Comparison (Adam)")
print("="*60)

lr_list    = [0.1, 0.01, 0.001]
lr_colors  = {0.1: 'tomato', 0.01: 'goldenrod', 0.001: 'steelblue'}
lr_results = {}

for lr in lr_list:
    print(f"\n--- Adam lr={lr} ---")
    m   = SimpleCNN().to(device)
    opt = optim.Adam(m.parameters(), lr=lr)
    tr_l, vl_l, tr_a, vl_a = train_model(m, opt, num_epochs=10, desc=f'lr={lr}')
    _, test_acc, _, _, _    = evaluate_model(m)
    lr_results[lr] = dict(train_losses=tr_l, val_losses=vl_l,
                          train_accs=tr_a,   val_accs=vl_a,
                          test_acc=test_acc)
    print(f"Adam lr={lr} test accuracy: {test_acc:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, title, key, ylabel in zip(axes.flat, titles, keys, ylabels):
    for lr, res in lr_results.items():
        ax.plot(epochs_range, res[key], '-o', label=f'lr={lr}',
                color=lr_colors[lr], markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_title(title);    ax.legend()

plt.suptitle('Task 3 – Learning Rate Comparison (Adam, MNIST)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('task3_lr_comparison.png', dpi=100)
plt.close()
print("saved: task3_lr_comparison.png")

print("\nTask 3 Summary:")
for lr, res in lr_results.items():
    print(f"  Adam lr={lr}: train_acc={res['train_accs'][-1]:.4f}, "
          f"val_acc={res['val_accs'][-1]:.4f}, test_acc={res['test_acc']:.4f}")

# ============================================================
# Task 4：卷积核可视化（第一层，16 个 3×3 卷积核）
# ============================================================
print("\n" + "="*60)
print("Task 4: Conv Kernel Visualization")
print("="*60)

conv1_weights = model.conv1.weight.data.cpu()   # [16, 1, 3, 3]

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    ax     = axes[i // 8, i % 8]
    kernel = conv1_weights[i, 0].numpy()
    vmax   = max(abs(kernel.min()), abs(kernel.max()))
    ax.imshow(kernel, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'K{i+1}', fontsize=9)
    ax.axis('off')

plt.suptitle('Task 4 – Conv1 Layer: 16 Learned 3×3 Kernels', fontsize=12)
plt.tight_layout()
plt.savefig('task4_conv_kernels.png', dpi=120)
plt.close()
print("saved: task4_conv_kernels.png")

# ============================================================
# Task 5：Feature Map 可视化（第一层卷积输出）
# ============================================================
print("\n" + "="*60)
print("Task 5: Feature Map Visualization")
print("="*60)

# 固定选取一张测试图（index=0），便于报告复现
fixed_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)
test_img_raw, test_label_val = next(iter(fixed_loader))
test_label_val = test_label_val.item()

model.eval()
with torch.no_grad():
    feat_after_conv1 = model.relu1(model.conv1(test_img_raw.to(device)))  # [1, 16, 28, 28]
    feat_maps = feat_after_conv1.cpu().squeeze(0)  # [16, 28, 28]

# 布局：第一列显示原图，其余 16 列显示 feature maps
fig = plt.figure(figsize=(19, 4))
# 原图
ax0 = fig.add_subplot(2, 9, 1)
ax0.imshow(test_img_raw.squeeze().numpy(), cmap='gray')
ax0.set_title(f'Input\nlabel: {test_label_val}', fontsize=9)
ax0.axis('off')
fig.add_subplot(2, 9, 10).axis('off')   # 左下角留空

for i in range(16):
    row = i // 8
    col = (i % 8) + 2   # 列偏移 2（第1列是原图）
    ax  = fig.add_subplot(2, 9, row * 9 + col)
    fm  = feat_maps[i].numpy()
    ax.imshow(fm, cmap='viridis')
    ax.set_title(f'FM {i+1}', fontsize=8)
    ax.axis('off')

plt.suptitle(f'Task 5 – Feature Maps after Conv1+ReLU (input label: {test_label_val})',
             fontsize=12)
plt.tight_layout()
plt.savefig('task5_feature_maps.png', dpi=120)
plt.close()
print("saved: task5_feature_maps.png")

# ============================================================
# Task 6：错误分类样本分析
# ============================================================
print("\n" + "="*60)
print("Task 6: Misclassified Samples Analysis")
print("="*60)

# 从 Task 1 的测试结果中找错误样本
wrong_imgs, wrong_trues, wrong_preds = [], [], []
for img, true_lbl, pred_lbl in zip(t1_all_imgs, t1_all_labels, t1_all_preds):
    if true_lbl != pred_lbl:
        wrong_imgs.append(img)
        wrong_trues.append(true_lbl)
        wrong_preds.append(pred_lbl)
    if len(wrong_imgs) >= 16:
        break

total_wrong = sum(p != l for p, l in zip(t1_all_preds, t1_all_labels))
print(f"Total misclassified: {total_wrong} / {len(t1_all_labels)}")
print("Showing first 16 misclassified samples...")

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i in range(16):
    ax  = axes[i // 8, i % 8]
    img = wrong_imgs[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'True: {wrong_trues[i]}\nPred: {wrong_preds[i]}', fontsize=9, color='red')
    ax.axis('off')

plt.suptitle(f'Task 6 – Misclassified Samples (total: {total_wrong}/{len(t1_all_labels)})',
             fontsize=12)
plt.tight_layout()
plt.savefig('task6_misclassified.png', dpi=120)
plt.close()
print("saved: task6_misclassified.png")

# 统计最容易混淆的类别对
from collections import Counter
confusion_pairs = Counter()
for true_lbl, pred_lbl in zip(t1_all_labels, t1_all_preds):
    if true_lbl != pred_lbl:
        confusion_pairs[(true_lbl, pred_lbl)] += 1

print("\nTop 10 most confused pairs (true→pred: count):")
for (t, p), cnt in confusion_pairs.most_common(10):
    print(f"  {t} → {p}: {cnt} times")

# ============================================================
# Task 7：混淆矩阵
# ============================================================
print("\n" + "="*60)
print("Task 7: Confusion Matrix")
print("="*60)

cm = confusion_matrix(t1_all_labels, t1_all_preds)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label',      fontsize=12)
ax.set_title('Task 7 – Confusion Matrix (MNIST Test Set)', fontsize=13)
plt.tight_layout()
plt.savefig('task7_confusion_matrix.png', dpi=120)
plt.close()
print("saved: task7_confusion_matrix.png")

# 找出最混淆的类别对
off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
max_idx  = np.unravel_index(off_diag.argmax(), off_diag.shape)
print(f"Most confused pair: true={max_idx[0]} predicted as {max_idx[1]} "
      f"({off_diag[max_idx]} times)")

# 每类准确率
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class accuracy:")
for cls, acc in enumerate(per_class_acc):
    print(f"  Digit {cls}: {acc:.4f}")

# ============================================================
# 总结输出
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Task 1 – SimpleCNN Adam lr=0.001 (10 ep): test acc = {t1_test_acc:.4f}")
print("\nTask 2 – Optimizer Comparison:")
for name, res in opt_results.items():
    print(f"  {name:15s}: test acc = {res['test_acc']:.4f}")
print("\nTask 3 – Learning Rate Comparison (Adam):")
for lr, res in lr_results.items():
    print(f"  lr={lr}: test acc = {res['test_acc']:.4f}")
print(f"\nTask 7 – Most confused pair: {max_idx[0]} → {max_idx[1]} ({off_diag[max_idx]} times)")
print("\nAll plots saved. Experiment complete.")
