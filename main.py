import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# 检查环境
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 张量操作测试
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print("tensor test:", a + b)

# ============================================================
# Task 2: 加载数据集（MNIST + CIFAR-10）
# ============================================================

# --- MNIST ---
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

# 划分训练集和验证集
train_size = 50000
val_size = len(mnist_train_full) - train_size
mnist_train, mnist_val = random_split(mnist_train_full, [train_size, val_size])

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

print(f"\nMNIST 训练集: {len(mnist_train)}, 验证集: {len(mnist_val)}, 测试集: {len(mnist_test)}")

# 显示8张样本图像
classes_mnist = [str(i) for i in range(10)]
imgs, labels = next(iter(DataLoader(mnist_train_full, batch_size=8, shuffle=True)))

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax = axes[i]
    img = imgs[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'label:{labels[i].item()}', fontsize=9)
    ax.axis('off')
fig.suptitle('MNIST sample images')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=100)
plt.close()
print("saved: mnist_samples.png")

# ============================================================
# Task 3: 定义CNN模型
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10, in_channels=1).to(device)
print("\nModel structure:")
print(model)

# ============================================================
# Task 4 & 5: 训练 + 验证
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    v_loss = 0
    v_correct = 0
    v_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, predicted = outputs.max(1)
            v_correct += predicted.eq(labels).sum().item()
            v_total += labels.size(0)

    val_loss = v_loss / len(val_loader)
    val_acc = v_correct / v_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
          f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

# ============================================================
# Task 6: 测试模型
# ============================================================

model.eval()
test_loss = 0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_imgs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_imgs.extend(imgs.cpu())

test_loss /= len(test_loader)
test_acc = test_correct / test_total
print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# 显示8张测试图像及预测结果
fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
for i in range(8):
    ax = axes[i]
    img = all_imgs[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    true = all_labels[i]
    pred = all_preds[i]
    color = 'green' if true == pred else 'red'
    ax.set_title(f'T:{true}\nP:{pred}', fontsize=9, color=color)
    ax.axis('off')
fig.suptitle('Test predictions (green=correct, red=wrong)')
plt.tight_layout()
plt.savefig('test_predictions.png', dpi=100)
plt.close()
print("saved: test_predictions.png")

# ============================================================
# Task 7: 绘制训练曲线
# ============================================================

epochs = range(1, num_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_losses, label='train loss')
ax1.plot(epochs, val_losses, label='val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss curve')
ax1.legend()

ax2.plot(epochs, train_accs, label='train acc')
ax2.plot(epochs, val_accs, label='val acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy curve')
ax2.legend()

plt.tight_layout()
plt.savefig('train_curve.png', dpi=100)
plt.close()
print("saved: train_curve.png")

# ============================================================
# Advanced Task 1: 修改网络结构（加dropout）
# ============================================================

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model2 = ImprovedCNN(num_classes=10, in_channels=1).to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

print("\n--- Advanced Task 1: ImprovedCNN (more filters + dropout) ---")
for epoch in range(5):
    model2.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer2.zero_grad()
        outputs = model2(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    print(f"Epoch [{epoch+1}/5] train loss: {total_loss/len(train_loader):.4f}, train acc: {correct/total:.4f}")

model2.eval()
adv1_correct = 0
adv1_total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model2(imgs)
        _, predicted = outputs.max(1)
        adv1_correct += predicted.eq(labels).sum().item()
        adv1_total += labels.size(0)
adv1_acc = adv1_correct / adv1_total
print(f"ImprovedCNN test accuracy: {adv1_acc:.4f}")
print(f"vs SimpleCNN: {test_acc:.4f}  -> delta: {adv1_acc - test_acc:+.4f}")

# ============================================================
# Advanced Task 2: 比较不同优化器（SGD vs Adam）
# ============================================================

print("\n--- Advanced Task 2: SGD vs Adam ---")

def train_eval(opt_name, lr):
    m = SimpleCNN(num_classes=10, in_channels=1).to(device)
    if opt_name == 'SGD':
        opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    else:
        opt = optim.Adam(m.parameters(), lr=lr)
    for epoch in range(5):
        m.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(m(imgs), labels)
            loss.backward()
            opt.step()
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, predicted = m(imgs).max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

sgd_acc = train_eval('SGD', lr=0.01)
adam_acc = train_eval('Adam', lr=0.001)
print(f"SGD  (lr=0.01):  test acc = {sgd_acc:.4f}")
print(f"Adam (lr=0.001): test acc = {adam_acc:.4f}")

# ============================================================
# Advanced Task 3: CIFAR-10
# ============================================================

print("\n--- Advanced Task 3: CIFAR-10 ---")

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

cifar_train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

c_train_size = 45000
c_val_size = len(cifar_train_full) - c_train_size
cifar_train, cifar_val = random_split(cifar_train_full, [c_train_size, c_val_size])

c_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
c_val_loader = DataLoader(cifar_val, batch_size=64, shuffle=False)
c_test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)

print(f"CIFAR-10 训练集: {len(cifar_train)}, 验证集: {len(cifar_val)}, 测试集: {len(cifar_test)}")

# 显示8张CIFAR-10样本
cifar_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
c_imgs, c_labels = next(iter(DataLoader(cifar_train_full, batch_size=8, shuffle=True)))

fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.247, 0.243, 0.261])
for i in range(8):
    ax = axes[i]
    img = c_imgs[i].numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(cifar_classes[c_labels[i].item()], fontsize=8)
    ax.axis('off')
fig.suptitle('CIFAR-10 sample images')
plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=100)
plt.close()
print("saved: cifar10_samples.png")

# CIFAR-10 CNN模型（3通道输入，输出尺寸8x8）
class CIFARCNN(nn.Module):
    def __init__(self):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

cifar_model = CIFARCNN().to(device)
cifar_opt = optim.Adam(cifar_model.parameters(), lr=0.001)

c_train_losses = []
c_val_losses = []
c_train_accs = []
c_val_accs = []

for epoch in range(10):
    cifar_model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in c_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        cifar_opt.zero_grad()
        outputs = cifar_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        cifar_opt.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    c_train_losses.append(total_loss / len(c_train_loader))
    c_train_accs.append(correct / total)

    cifar_model.eval()
    v_loss = 0
    v_correct = 0
    v_total = 0
    with torch.no_grad():
        for imgs, labels in c_val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = cifar_model(imgs)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, predicted = outputs.max(1)
            v_correct += predicted.eq(labels).sum().item()
            v_total += labels.size(0)

    c_val_losses.append(v_loss / len(c_val_loader))
    c_val_accs.append(v_correct / v_total)
    print(f"CIFAR Epoch [{epoch+1}/10] "
          f"train loss: {c_train_losses[-1]:.4f}, train acc: {c_train_accs[-1]:.4f} | "
          f"val loss: {c_val_losses[-1]:.4f}, val acc: {c_val_accs[-1]:.4f}")

# CIFAR-10 测试
cifar_model.eval()
c_correct = 0
c_total = 0
c_all_preds = []
c_all_labels = []
c_all_imgs = []
with torch.no_grad():
    for imgs, labels in c_test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = cifar_model(imgs)
        _, predicted = outputs.max(1)
        c_correct += predicted.eq(labels).sum().item()
        c_total += labels.size(0)
        c_all_preds.extend(predicted.cpu().numpy())
        c_all_labels.extend(labels.cpu().numpy())
        c_all_imgs.extend(imgs.cpu())

cifar_test_acc = c_correct / c_total
print(f"CIFAR-10 test accuracy: {cifar_test_acc:.4f}")

# CIFAR-10 训练曲线
epochs = range(1, 11)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, c_train_losses, label='train loss')
ax1.plot(epochs, c_val_losses, label='val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('CIFAR-10 Loss curve')
ax1.legend()

ax2.plot(epochs, c_train_accs, label='train acc')
ax2.plot(epochs, c_val_accs, label='val acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('CIFAR-10 Accuracy curve')
ax2.legend()

plt.tight_layout()
plt.savefig('cifar10_curve.png', dpi=100)
plt.close()
print("saved: cifar10_curve.png")

# CIFAR-10 测试预测图
fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
for i in range(8):
    ax = axes[i]
    img = c_all_imgs[i].numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    true = cifar_classes[c_all_labels[i]]
    pred = cifar_classes[c_all_preds[i]]
    color = 'green' if c_all_labels[i] == c_all_preds[i] else 'red'
    ax.set_title(f'T:{true}\nP:{pred}', fontsize=7, color=color)
    ax.axis('off')
fig.suptitle('CIFAR-10 test predictions')
plt.tight_layout()
plt.savefig('cifar10_predictions.png', dpi=100)
plt.close()
print("saved: cifar10_predictions.png")

print("\n========== Summary ==========")
print(f"MNIST  SimpleCNN   test acc: {test_acc:.4f}")
print(f"MNIST  ImprovedCNN test acc: {adv1_acc:.4f}")
print(f"MNIST  SGD  (lr=0.01)  acc: {sgd_acc:.4f}")
print(f"MNIST  Adam (lr=0.001) acc: {adam_acc:.4f}")
print(f"CIFAR-10 CNN          acc: {cifar_test_acc:.4f}")
print("All done.")
