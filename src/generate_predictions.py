import os
import joblib
import pandas as pd

DATA_FILE = "data/processed/player_salary_efficiency.csv"
MODEL_FILE = "artifacts/model/mp_value_ridge_pipeline.joblib"
ROLES_FILE = "artifacts/clusters/player_roles.csv"
OUT_FILE = "data/processed/player_predictions.csv"

def main():
    df = pd.read_csv(DATA_FILE)

    bundle = joblib.load(MODEL_FILE)
    pipeline = bundle["pipeline"]
    features = bundle["features"]

    present_features = []
    for col in features:
        if col in df.columns:
            present_features.append(col)

    X = df[present_features]
    preds = pipeline.predict(X)
    df["pred_mp_value"] = preds

    if os.path.exists(ROLES_FILE):
        roles = pd.read_csv(ROLES_FILE)
        cols = ["Name", "team", "role_cluster"]
        roles = roles[cols]
        df = df.merge(roles, on=["Name", "team"], how="left")

    df.to_csv(OUT_FILE, index=False)
    print("Saved ->", OUT_FILE)

if __name__ == "__main__":
    main()
