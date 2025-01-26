import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Best hyperparameters from tuning
BEST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 10,
    "min_samples_split": 3,
}

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model with best hyperparameters
model = RandomForestClassifier(
    **BEST_PARAMS,
    random_state=42
)
model.fit(X_train, y_train)

# Save the model
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as models/iris_model.pkl")
