from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0  # Normalize pixel values
y = y.astype('int')

# 2. Split for training (optional, small subset for speed)
X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, random_state=42)

# 3. Train a simple logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 4. Save the model
joblib.dump(clf, 'mnist_model.pkl')

print("✅ Model trained and saved as mnist_model.pkl")
