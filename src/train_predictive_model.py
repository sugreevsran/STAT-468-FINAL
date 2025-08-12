import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

IN_FILE = "data/processed/player_salary_efficiency.csv"
OUT_DIR = "artifacts/model"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "mp_value"

NUMERIC_FEATURES = [
    "games_played", "cap_hit", "onIce_corsiPercentage", "icetime_minutes",
    "I_F_points", "I_F_goals", "I_F_primaryAssists", "I_F_xGoals",
    "I_F_takeaways", "I_F_giveaways",
    "I_F_goals_per60", "I_F_primaryAssists_per60", "I_F_secondaryAssists_per60",
    "I_F_points_per60", "I_F_shotsOnGoal_per60", "I_F_xGoals_per60",
    "I_F_hits_per60", "I_F_takeaways_per60", "I_F_giveaways_per60"
]

CATEGORICAL_FEATURES = ["position"]

def main():
    df = pd.read_csv(IN_FILE)
    df = df[df[TARGET].notna()]
    df = df.replace([np.inf, -np.inf], np.nan)

    available_numeric = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            available_numeric.append(col)

    available_categorical = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            available_categorical.append(col)

    X = df[available_numeric + available_categorical].copy()
    y = df[TARGET].astype(float)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, available_numeric),
        ("cat", categorical_transformer, available_categorical)
    ])

    ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", ridge_model)
    ])

    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "r2": make_scorer(r2_score),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False)
    }

    results = cross_validate(
        pipe, X, y, cv=cv_splitter, scoring=scoring, return_estimator=True
    )

    best_index = int(np.argmax(results["test_r2"]))
    best_alpha = float(results["estimator"][best_index].named_steps["model"].alpha_)

    pipe.fit(X, y)

    model_bundle = {
        "pipeline": pipe,
        "features": available_numeric + available_categorical,
        "target": TARGET
    }
    joblib.dump(model_bundle, os.path.join(OUT_DIR, "mp_value_ridge_pipeline.joblib"))

    metrics_summary = {
        "n": int(len(df)),
        "p": int(len(available_numeric) + len(available_categorical)),
        "cv_r2_mean": float(np.mean(results["test_r2"])),
        "cv_r2_std": float(np.std(results["test_r2"])),
        "cv_mae_mean": float(-np.mean(results["test_mae"])),
        "cv_mae_std": float(np.std(-results["test_mae"])),
        "best_alpha": best_alpha
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("Saved model ->", os.path.join(OUT_DIR, "mp_value_ridge_pipeline.joblib"))
    print("CV summary:", metrics_summary)

if __name__ == "__main__":
    main()
