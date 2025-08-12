import os
import re
import unicodedata
import pandas as pd

MP_FILE = "data/processed/moneypuck_clean.csv"
SAL_FILE = "data/processed/puckpedia_salaries.csv"
OUT_MERGED_FILE = "data/processed/player_data.csv"
OUT_EFF_FILE = "data/processed/player_salary_efficiency.csv"
UNMATCHED_FILE = "data/processed/unmatched_names.csv"

MIN_POINTS = 30
MIN_GP = 41

ALIASES = {"MITCHELL": "MITCH"}

def normalize(name):
    if pd.isna(name):
        return ""
    text = str(name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.upper().strip()
    parts = text.split()
    if len(parts) > 0:
        first = parts[0]
        if first in ALIASES:
            parts[0] = ALIASES[first]
    return " ".join(parts)

def nz_div(num, den):
    if pd.isna(den):
        return pd.NA
    if den == 0 or den == 0.0:
        return pd.NA
    return num / den

def safe_num(series_like, col):
    if col in series_like.columns:
        return pd.to_numeric(series_like[col], errors="coerce")
    else:
        return pd.Series(pd.NA, index=series_like.index)

def main():
    os.makedirs(os.path.dirname(OUT_MERGED_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_EFF_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(UNMATCHED_FILE), exist_ok=True)

    stats = pd.read_csv(MP_FILE)
    stats = stats.rename(columns={"name": "Name"})
    sal = pd.read_csv(SAL_FILE)

    stats["Name_norm"] = stats["Name"].apply(normalize)
    sal["Name_norm"] = sal["Name"].apply(normalize)

    if "Cap Hit" in sal.columns and "CapHit" not in sal.columns:
        sal = sal.rename(columns={"Cap Hit": "CapHit"})

    sal = sal.sort_values("CapHit", ascending=False)
    sal = sal.drop_duplicates(subset=["Name_norm"], keep="first")

    cols_to_keep = ["Name_norm", "CapHit", "Pos", "Length", "Start Year"]
    sal_small = sal[cols_to_keep]
    merged = stats.merge(sal_small, on="Name_norm", how="left")

    merged.to_csv(OUT_MERGED_FILE, index=False)

    if "CapHit" in merged.columns:
        unmatched = merged[merged["CapHit"].isna()]
        cols2 = ["Name", "team", "position", "games_played"]
        cols2 = [c for c in cols2 if c in unmatched.columns]
        unmatched = unmatched[cols2].drop_duplicates()
        unmatched.to_csv(UNMATCHED_FILE, index=False)
    else:
        empty_df = pd.DataFrame(columns=["Name", "team", "position", "games_played"])
        empty_df.to_csv(UNMATCHED_FILE, index=False)

    df = merged.copy()

    if "CapHit" in df.columns:
        df["cap_hit"] = pd.to_numeric(df["CapHit"], errors="coerce")
    else:
        df["cap_hit"] = pd.NA

    df = df[df["cap_hit"].notna()]
    df = df[df["cap_hit"] > 0]
    df = df.copy()

    df["cap_millions"] = df["cap_hit"] / 1000000.0

    keep_numeric = [
        "I_F_points", "I_F_goals", "I_F_primaryAssists", "I_F_xGoals",
        "I_F_takeaways", "I_F_giveaways", "onIce_corsiPercentage",
        "mp_value", "gs_per_game", "gs_per60", "icetime_minutes",
        "I_F_goals_per60", "I_F_primaryAssists_per60", "I_F_secondaryAssists_per60",
        "I_F_points_per60", "I_F_shotsOnGoal_per60", "I_F_xGoals_per60",
        "I_F_hits_per60", "I_F_takeaways_per60", "I_F_giveaways_per60",
    ]

    for col in keep_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    if ("I_F_points" in df.columns) and ("games_played" in df.columns):
        before = len(df)
        mask_points = pd.to_numeric(df["I_F_points"], errors="coerce") >= MIN_POINTS
        mask_gp = pd.to_numeric(df["games_played"], errors="coerce") >= MIN_GP
        df = df[mask_points & mask_gp].copy()
        print("Filtered for contributors (>= {} points & >= {} GP): {} -> {} rows".format(
            MIN_POINTS, MIN_GP, before, len(df))
        )

    cost_per_point = []
    cost_per_goal = []
    cost_per_primary_assist = []
    cost_per_xgoal = []
    net_takeaway_value = []
    possession_impact_index = []
    cost_per_corsi_above_50 = []
    cost_per_mp_value = []

    for i in range(len(df)):
        ch = df.iloc[i]["cap_hit"]
        pts = df.iloc[i]["I_F_points"]
        goals = df.iloc[i]["I_F_goals"]
        pas = df.iloc[i]["I_F_primaryAssists"]
        xg = df.iloc[i]["I_F_xGoals"]
        take = df.iloc[i]["I_F_takeaways"]
        give = df.iloc[i]["I_F_giveaways"]
        corsi = df.iloc[i]["onIce_corsiPercentage"] if "onIce_corsiPercentage" in df.columns else pd.NA
        mpv = df.iloc[i]["mp_value"] if "mp_value" in df.columns else pd.NA

        cost_per_point.append(nz_div(ch, pts))
        cost_per_goal.append(nz_div(ch, goals))
        cost_per_primary_assist.append(nz_div(ch, pas))
        cost_per_xgoal.append(nz_div(ch, xg))

        net_takeaway_value.append(nz_div(take - give, ch))

        give_for_index = give
        if give_for_index == 0:
            give_for_index = 1
        possession_impact_index.append(nz_div(take / give_for_index, ch))

        if pd.notna(corsi):
            above50 = corsi - 50
            if above50 < 0:
                above50 = 0
            cost_per_corsi_above_50.append(nz_div(ch, above50))
        else:
            cost_per_corsi_above_50.append(pd.NA)

        cost_per_mp_value.append(nz_div(ch, mpv))

    df["cost_per_point"] = cost_per_point
    df["cost_per_goal"] = cost_per_goal
    df["cost_per_primary_assist"] = cost_per_primary_assist
    df["cost_per_xgoal"] = cost_per_xgoal
    df["net_takeaway_value"] = net_takeaway_value
    df["possession_impact_index"] = possession_impact_index
    df["cost_per_corsi_above_50"] = cost_per_corsi_above_50
    df["cost_per_mp_value"] = cost_per_mp_value

    keep = [
        "Name", "team", "position", "season", "games_played",
        "cap_hit", "cap_millions",
        "I_F_points", "I_F_goals", "I_F_primaryAssists", "I_F_xGoals",
        "I_F_takeaways", "I_F_giveaways", "onIce_corsiPercentage",
        "mp_value", "gs_per_game", "gs_per60", "icetime_minutes",
        "I_F_goals_per60", "I_F_primaryAssists_per60", "I_F_secondaryAssists_per60",
        "I_F_points_per60", "I_F_shotsOnGoal_per60", "I_F_xGoals_per60",
        "I_F_hits_per60", "I_F_takeaways_per60", "I_F_giveaways_per60",
        "cost_per_point", "cost_per_goal", "cost_per_primary_assist", "cost_per_xgoal",
        "net_takeaway_value", "possession_impact_index", "cost_per_corsi_above_50",
        "cost_per_mp_value",
    ]

    final_cols = []
    for c in keep:
        if c in df.columns:
            final_cols.append(c)

    eff = df[final_cols].copy()
    eff.to_csv(OUT_EFF_FILE, index=False)

    if "CapHit" in merged.columns:
        match_rate = merged["CapHit"].notna().mean() * 100
    else:
        match_rate = 0.0

    print("Saved base ->", OUT_MERGED_FILE)
    print("Saved efficiency ->", OUT_EFF_FILE)
    print("Salary match rate: {:.1f}%".format(match_rate))
    print("Unmatched list ->", UNMATCHED_FILE)

if __name__ == "__main__":
    main()
