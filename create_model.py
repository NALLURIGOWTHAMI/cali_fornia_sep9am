from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load California Housing dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Column names
columns = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Save model
obj = {
    "model": model,
    "columns": columns
}

joblib.dump(obj, "california.joblib")

print("Model created successfully!")
print("Columns:", columns)
print("Model saved as california.joblib")