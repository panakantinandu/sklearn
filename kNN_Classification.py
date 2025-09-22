# KNN_Classification.py

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend (saves images instead of showing)

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# 1. Load the dataset (only first 2 features: sepal length & sepal width)
iris = datasets.load_iris()
X = iris.data[:, :2]   # only sepal length and sepal width
y = iris.target

# Step size in the mesh
h = 0.02  

# 2. Define k values to test
k_values = [1, 3, 5, 10]

# 3. Plot decision boundaries for each k
plt.figure(figsize=(12, 10))

for i, k in enumerate(k_values, 1):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.subplot(2, 2, i)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=40)
    plt.title(f"k = {k}")

plt.tight_layout()

# Instead of plt.show(), save the figure
plt.savefig("knn_boundaries.png")
print("✅ Plot saved as knn_boundaries.png")
