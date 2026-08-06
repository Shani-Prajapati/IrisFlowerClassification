"""
CodeAlpha - Task 1: Iris Flower Classification
Trains several classifiers, compares them with cross-validation + a held-out
test set, tunes the best one with GridSearchCV, and saves the final pipeline
(scaler + model) to disk for the API/UI to use.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"

X = df[FEATURES].values
y_raw = df[TARGET].values

le = LabelEncoder()
y = le.fit_transform(y_raw)  # setosa=0/1/2 alphabetically
class_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------
# 2. Compare candidate models with 5-fold cross-validation on the
#    training set (keeps the test set completely untouched)
# ---------------------------------------------------------------------
candidates = {
    "Logistic Regression": Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    ),
    "KNN": Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
    ),
    "Decision Tree": Pipeline(
        [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]
    ),
    "Random Forest": Pipeline(
        [("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]
    ),
    "SVM (RBF)": Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))]
    ),
}

print("=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS (on training data)")
print("=" * 60)
cv_results = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores.mean()
    print(f"{name:22s}: {scores.mean()*100:.2f}%  (+/- {scores.std()*100:.2f}%)")

best_name = max(cv_results, key=cv_results.get)
print(f"\nBest candidate by CV: {best_name}")

# ---------------------------------------------------------------------
# 3. Hyperparameter tuning for SVM (typically the strongest + most
#    robust classifier on this dataset) via GridSearchCV
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (SVM)")
print("=" * 60)

svm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))])

param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.01, 0.1, 1],
    "clf__kernel": ["rbf", "linear", "poly"],
}

grid = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_*100:.2f}%")

best_model = grid.best_estimator_

# ---------------------------------------------------------------------
# 4. Evaluation on the held-out test set (30 samples -> a bit noisy,
#    since one wrong prediction swings it by 3.3%)
# ---------------------------------------------------------------------
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="macro")

# More reliable headline number: 10-fold CV accuracy on the FULL
# dataset using the tuned hyperparameters (averages out split luck).
final_pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE, **{k.replace("clf__", ""): v for k, v in grid.best_params_.items()}))]
)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
full_cv_scores = cross_val_score(final_pipe, X, y, cv=cv10, scoring="accuracy")
full_cv_acc = full_cv_scores.mean()

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)
print(f"Model: Tuned SVM ({grid.best_params_['clf__kernel']} kernel)")
print(f"10-fold CV Accuracy (full dataset): {full_cv_acc*100:.2f}%  (+/- {full_cv_scores.std()*100:.2f}%)")
print(f"Held-out Test Accuracy (30 samples): {test_acc*100:.2f}%  |  Macro F1: {test_f1*100:.2f}%")
print("\nClassification Report (held-out test set):")
print(classification_report(y_test, y_pred, target_names=class_names))
print("Confusion Matrix (held-out test set):")
print(confusion_matrix(y_test, y_pred))

# Refit best model on ALL data for the deployed model (standard practice
# once architecture + hyperparameters are locked in via CV)
best_model.fit(X, y)

# ---------------------------------------------------------------------
# 5. Persist model + metadata
# ---------------------------------------------------------------------
joblib.dump(best_model, "iris_model.pkl")
joblib.dump(le, "label_encoder.pkl")

metadata = {
    "features": FEATURES,
    "classes": class_names,
    "cv_accuracy_10fold": round(full_cv_acc * 100, 2),
    "held_out_test_accuracy": round(test_acc * 100, 2),
    "held_out_test_macro_f1": round(test_f1 * 100, 2),
    "best_params": grid.best_params_,
    "model_comparison": {k: round(v * 100, 2) for k, v in cv_results.items()},
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: iris_model.pkl, label_encoder.pkl, model_metadata.json")
