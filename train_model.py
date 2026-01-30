import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/transactions.csv")

# Categorical columns to encode
categorical_cols = ["txn_type", "merchant_category"]

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Features and target
X = data.drop(columns=["is_fraud", "card_id"])  # drop card_id and target
y = data["is_fraud"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Save model, scaler, and columns
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model trained successfully")
