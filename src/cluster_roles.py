import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

IN_FILE = "data/processed/player_salary_efficiency.csv"
OUT_DIR = "artifacts/clusters"
OUT_ROLES = os.path.join(OUT_DIR, "player_roles.csv")
OUT_ARTIFACTS = os.path.join(OUT_DIR, "cluster_artifacts.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_COLS = [
    "I_F_goals_per60",
    "I_F_primaryAssists_per60",
    "I_F_secondaryAssists_per60",
    "I_F_points_per60",
    "I_F_shotsOnGoal_per60",
    "I_F_xGoals_per60",
    "I_F_hits_per60",
    "I_F_takeaways_per60",
    "I_F_giveaways_per60",
]

def main(k=6):
    df = pd.read_csv(IN_FILE)
    mat = df[CLUSTER_COLS].copy()
    mat = mat.replace([np.inf, -np.inf], np.nan)
    for col in CLUSTER_COLS:
        col_median = mat[col].median()
        mat[col] = mat[col].fillna(col_median)
    mat = mat.fillna(0.0)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(mat)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xz)

    out = df[["Name", "team", "position", "cap_hit", "mp_value"]].copy()
    out["role_cluster"] = labels

    out.to_csv(OUT_ROLES, index=False)
    artifacts = {"scaler": scaler, "kmeans": kmeans, "columns": CLUSTER_COLS}
    joblib.dump(artifacts, OUT_ARTIFACTS)

    print("Saved clustered player roles to:", OUT_ROLES)
    print("Saved model artifacts to:", OUT_ARTIFACTS)

if __name__ == "__main__":
    main()
