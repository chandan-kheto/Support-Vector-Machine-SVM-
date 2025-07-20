# imort Libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Let's simplify to binary classification (2 classes only)
X = X[y != 2]
y = y[y != 2]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train SVM
model = SVC(kernel='linear')  # linear SVM
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Confusion Matrix
print("y_test:", y_test.tolist())
print("y_pred:", y_pred.tolist())
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
