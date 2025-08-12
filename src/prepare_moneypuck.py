import os
import pandas as pd
import numpy as np

RAW_FILE = "data/raw/skaters.csv"
OUT_FILE = "data/processed/moneypuck_clean.csv"

BASE_COLS = [
    "playerId", "season", "name", "team", "position", "situation",
    "games_played", "icetime", "shifts",
    "ppTimeOnIce", "pkTimeOnIce", "powerPlayIcetime", "shortHandedIcetime",
    "pp_toi", "pk_toi",
    "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_points",
    "I_F_shotsOnGoal", "I_F_xGoals", "I_F_hits", "I_F_takeaways", "I_F_giveaways",
    "onIce_xGoalsPercentage", "onIce_corsiPercentage", "onIce_fenwickPercentage",
    "gameScore"
]

RATE_COLS = [
    "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_points",
    "I_F_shotsOnGoal", "I_F_xGoals", "I_F_hits", "I_F_takeaways", "I_F_giveaways"
]

def rate_per60(counts, minutes):
    result = np.where((minutes > 0) & minutes.notna(), counts / minutes, 0.0)
    return result

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    if "situation" in df.columns:
        df = df[df["situation"] == "all"].copy()
        df.drop(columns=["situation"], inplace=True, errors="ignore")

    for col in BASE_COLS:
        if col not in df.columns:
            if col.startswith("I_F_") or col in ["shifts", "ppTimeOnIce", "pkTimeOnIce",
                                                 "powerPlayIcetime", "shortHandedIcetime",
                                                 "pp_toi", "pk_toi", "gameScore"]:
                df[col] = 0
            elif col in ["onIce_xGoalsPercentage", "onIce_corsiPercentage", "onIce_fenwickPercentage"]:
                df[col] = 50.0
            elif col in ["games_played", "icetime"]:
                df[col] = 0
            else:
                df[col] = pd.NA

    final_cols = []
    for col in BASE_COLS:
        if col in df.columns:
            final_cols.append(col)
    df = df[final_cols].copy()

    df["icetime_minutes"] = pd.to_numeric(df["icetime"], errors="coerce") / 60.0
    df.loc[df["icetime_minutes"] < 0, "icetime_minutes"] = np.nan

    gp = pd.to_numeric(df["games_played"], errors="coerce").replace({0: np.nan})
    gs = pd.to_numeric(df["gameScore"], errors="coerce").fillna(0.0)

    df["gs_per_game"] = (gs / gp).fillna(0.0)
    df["gs_per60"] = rate_per60(gs, df["icetime_minutes"])
    df["mp_value"] = df["gs_per60"]

    for col in RATE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        per60_col = col + "_per60"
        df[per60_col] = rate_per60(df[col], df["icetime_minutes"])

    df["season"] = pd.to_numeric(df["season"], errors="ignore")
    df["position"] = df["position"].astype(str)

    df.to_csv(OUT_FILE, index=False)

    print("Saved", OUT_FILE, "with", df.shape[0], "rows and", df.shape[1], "columns")
    print("Per-60 columns:", [c for c in df.columns if c.endswith("_per60")])

if __name__ == "__main__":
    main()
