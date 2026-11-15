
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD THE PROCESSED FOOTBALL DATA
# ==========================================

project_folder = Path(__file__).resolve().parent.parent

data_file = (
    project_folder / "data" / "processed"
    / "matches_features.csv"
)

models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"

models_folder.mkdir(parents=True, exist_ok=True)
outputs_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)

df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

df = df.sort_values("DateTime").reset_index(drop=True)

print("\n========== MODEL TRAINING ==========")
print("Total matches:", len(df))


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

target = "FTR"

categorical_features = ["HomeTeam", "AwayTeam"]

numerical_features = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_avg_points",
    "home_win_rate",
    "home_matches_available",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_avg_points",
    "away_win_rate",
    "away_matches_available"
]

X = df[categorical_features + numerical_features]
y = df[target]

# ==========================================
# 3. CHRONOLOGICAL DATA SPLIT
# 70% training, 15% validation, 15% testing
# ==========================================

total = len(df)

train_end = int(total * 0.70)
validation_end = int(total * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\nDATA SPLIT")
print("Training matches:", len(X_train))
print("Validation matches:", len(X_validation))
print("Testing matches:", len(X_test))


# ==========================================
# 4. PREPARE DATA PROCESSING
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 5. DEFINE OUR THREE ALGORITHMS
# ==========================================

algorithms = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 6. TRAIN AND COMPARE MODELS
# ==========================================

validation_results = []
trained_models = {}

print("\n========== VALIDATION RESULTS ==========")

for name, algorithm in algorithms.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)

    macro_f1 = f1_score(
        y_validation,
        predictions,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0
    )

    validation_results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))


# ==========================================
# 7. SELECT THE BEST VALIDATION MODEL
# ==========================================

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    "Macro_F1", ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Algorithm"]

print("\nBest validation model:", best_name)


# ==========================================
# 8. RETRAIN SELECTED MODEL
# USING TRAINING + VALIDATION DATA
# ==========================================

X_train_validation = pd.concat(
    [X_train, X_validation]
)

y_train_validation = pd.concat(
    [y_train, y_validation]
)

best_model = trained_models[best_name]

best_model.fit(X_train_validation, y_train_validation)


# ==========================================
# 9. FINAL EVALUATION ON UNSEEN TEST DATA
# ==========================================

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    labels=["H", "D", "A"],
    average="macro",
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")

print("Selected algorithm:", best_name)
print("Test accuracy:", round(test_accuracy, 4))
print("Test macro F1:", round(test_macro_f1, 4))

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    test_predictions,
    labels=["H", "D", "A"],
    target_names=["Home Win", "Draw", "Away Win"],
    zero_division=0
))


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=["H", "D", "A"]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual H", "Actual D", "Actual A"],
    columns=["Predicted H", "Predicted D", "Predicted A"]
)

print("\nCONFUSION MATRIX")
print(matrix_df)

matrix_df.to_csv(
    outputs_folder / "confusion_matrix.csv"
)


# ==========================================
# 11. SAVE MODEL AND RESULTS
# ==========================================

joblib.dump(
    best_model,
    models_folder / "best_model.joblib"
)

results_df.to_csv(
    outputs_folder / "model_comparison.csv",
    index=False
)

pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions
}).to_csv(
    outputs_folder / "test_predictions.csv",
    index=False
)

print("\nSaved selected model:", models_folder / "best_model.joblib")
print("Saved evaluation results in outputs/")

print("\n========== TRAINING COMPLETE ==========")
