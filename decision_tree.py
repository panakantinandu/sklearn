from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Train and evaluate trees with different depths
depths = [1, 2, 3]
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)

    # Training accuracy
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    # Test accuracy
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"Depth = {d}: Training Accuracy = {train_acc:.3f}, Test Accuracy = {test_acc:.3f}")
