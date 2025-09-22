# kNN Classifier Performance Evaluation (Iris Dataset)
# Uses non-GUI backend to avoid Tkinter errors

import matplotlib
matplotlib.use("PDF")  # Use PDF backend instead of TkAgg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train kNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix (k=5, Iris)")
plt.savefig("confusion_matrix.pdf")  # Save to file
print("\nConfusion matrix saved as 'confusion_matrix.pdf'")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ROC Curve and AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = knn.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(6, 5))
colors = ["blue", "green", "red"]
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{iris.target_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (k=5, Iris)")
plt.legend(loc="lower right")
plt.savefig("roc_curve.pdf")  # Save to file
print("ROC curve saved as 'roc_curve.pdf'")
