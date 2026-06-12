import numpy as np
import matplotlib
matplotlib.use('Agg')  # save to file (no display needed)
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ======================================================
# Task 1: Data Preparation
# ======================================================
digits = load_digits()
X = digits.data      # (1797, 64)
y = digits.target    # (1797,)
images = digits.images  # (1797, 8, 8)

print("=" * 55)
print("Task 1: Data Preparation")
print(f"  Total samples      : {len(images)}")
print(f"  Image size         : {images.shape[1:]}  (H x W)")
print(f"  Class labels       : {np.unique(y)}")
print(f"  Feature vector dim : {X.shape[1]}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"label: {y[i]}")
    ax.axis('off')
fig.suptitle("Sample images from sklearn digits dataset (with true labels)")
fig.tight_layout()
fig.savefig("task1_samples.png", dpi=100)
plt.close(fig)
print("  Saved: task1_samples.png")

# ======================================================
# Task 2: Data Splitting
# ======================================================
print("\n" + "=" * 55)
print("Task 2: Data Splitting")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")
print(f"  Test ratio   : {X_test.shape[0] / len(X):.1%}")

# ======================================================
# Task 3: Feature Representation
# ======================================================
print("\n" + "=" * 55)
print("Task 3: Feature Representation")
print(f"  Original image shape : {images[0].shape}  (8 x 8 pixels)")
print(f"  Flattened vector     : {X[0].shape}  (64-dim)")
print(f"  Sample[0] first 10 values : {X[0][:10].astype(int)}")
print("  (Each pixel's grayscale value [0-16] becomes one feature)")

# Visualise the flatten process for one sample
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(images[0], cmap='gray')
axes[0].set_title("Original 8x8 image")
axes[0].axis('off')
axes[1].bar(range(64), X[0], color='steelblue')
axes[1].set_title("Flattened 64-dim feature vector")
axes[1].set_xlabel("Feature index (pixel position)")
axes[1].set_ylabel("Pixel value (0-16)")
fig.tight_layout()
fig.savefig("task3_flatten.png", dpi=100)
plt.close(fig)
print("  Saved: task3_flatten.png")

# ======================================================
# Task 4: Model Training
# ======================================================
print("\n" + "=" * 55)
print("Task 4: Model Training")

models = {
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "SVM (RBF)":           SVC(kernel='rbf', C=10, gamma=0.001, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name:<22} : {acc:.4f}  ({acc:.1%})")

# ======================================================
# Task 5: Results Comparison
# ======================================================
print("\n" + "=" * 55)
print("Task 5: Results Comparison")

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\n  {'Rank':<5} {'Model':<24} {'Test Accuracy':>14}")
print("  " + "-" * 45)
for rank, (name, acc) in enumerate(sorted_results, 1):
    print(f"  {rank:<5} {name:<24} {acc:>13.4f}")

best_name, best_acc   = sorted_results[0]
worst_name, worst_acc = sorted_results[-1]
print(f"\n  Best  : {best_name} ({best_acc:.4f})")
print(f"  Worst : {worst_name} ({worst_acc:.4f})")
print(f"  Gap   : {best_acc - worst_acc:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
names = [n for n, _ in sorted_results]
accs  = [a for _, a in sorted_results]
colors = ['gold' if n == best_name else 'salmon' if n == worst_name
          else 'steelblue' for n in names]
bars = ax.bar(names, accs, color=colors)
ax.set_ylim(max(0, min(accs) - 0.05), 1.02)
ax.set_ylabel("Test Accuracy")
ax.set_title("Model Accuracy Comparison — sklearn Digits Dataset")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{acc:.4f}", ha='center', va='bottom', fontsize=9)
ax.tick_params(axis='x', rotation=15)
fig.tight_layout()
fig.savefig("task5_accuracy_comparison.png", dpi=100)
plt.close(fig)
print("  Saved: task5_accuracy_comparison.png")

# ======================================================
# Task 6: Error Analysis
# ======================================================
print("\n" + "=" * 55)
print("Task 6: Error Analysis")
print(f"  (Using best model: {best_name})")

y_pred_best = predictions[best_name]
errors = np.where(y_test != y_pred_best)[0]
print(f"  Misclassified : {len(errors)} / {len(y_test)} samples")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=digits.target_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f"Confusion Matrix — {best_name}")
fig.tight_layout()
fig.savefig("task6_confusion_matrix.png", dpi=100)
plt.close(fig)
print("  Saved: task6_confusion_matrix.png")

# Misclassified sample grid
n_show = min(10, len(errors))
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i < n_show:
        idx = errors[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f"True:{y_test[idx]}  Pred:{y_pred_best[idx]}", fontsize=9)
    ax.axis('off')
fig.suptitle(f"Misclassified Samples — {best_name}")
fig.tight_layout()
fig.savefig("task6_misclassified.png", dpi=100)
plt.close(fig)
print("  Saved: task6_misclassified.png")

# Top confused digit pairs
off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
print("\n  Top 5 confused pairs (true → predicted, count):")
for _ in range(5):
    idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
    if off_diag[idx] == 0:
        break
    print(f"    {idx[0]} → {idx[1]} : {int(off_diag[idx])} times")
    off_diag[idx] = 0

print("\n  All done. PNG files saved in current directory.")
