import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


student_id = "26L-9001"

data_file = "data/breast_cancer.csv"
model_file = "model/breast_cancer_model_26L-9001.pkl"


data = pd.read_csv(data_file)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


data = data.replace(["?", "NA", "N/A", ""], pd.NA)

if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)

if "id" in data.columns:
    data = data.drop("id", axis=1)


if "diagnosis" in data.columns:
    target = "diagnosis"
else:
    target = "target"


X = data.drop(target, axis=1)
y = data[target]


valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]


if y.dtype == "object":
    y = y.astype(str).str.strip()

    if set(y.unique()).issubset({"B", "M"}):
        y = y.map({"B": 0, "M": 1})
    else:
        y = pd.factorize(y)[0]


for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")


empty_columns = X.columns[X.isna().all()]

if len(empty_columns) > 0:
    X = X.drop(columns=empty_columns)


X = X.fillna(X.median())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing completed.")


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Student ID:", student_id)
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


os.makedirs("model", exist_ok=True)


model_data = {
    "model": model,
    "scaler": scaler,
    "features": X.columns.tolist(),
    "student_id": student_id
}

joblib.dump(model_data, model_file)

print("\nModel saved successfully.")
print("Location:", model_file)