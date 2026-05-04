import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

print("Loading dataset...")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/diabetes.csv")

# -------------------------------
# Data Cleaning
# -------------------------------
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, df[col].median())

print("Data cleaned")

# 

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data splitted")

# -------------------------------
# Models
# -------------------------------
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )
}

best_acc = 0
best_model = None
best_name = ""

print("\nTraining all models...\n")

for name, model in models.items():

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(name, "Accuracy =", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = pipe
        best_name = name

# -------------------------------
# Save model

joblib.dump(best_model, "diabetes_xai_scaled_model1.pkl")

print("\nBest Model:", best_name)
print("Best Accuracy:", best_acc)
print("Model saved successfully!")