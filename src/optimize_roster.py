import duckdb
import pandas as pd
from ortools.sat.python import cp_model

S3_URL = "https://stat468-final-project.s3.us-east-1.amazonaws.com/player_predictions.csv"

CAP = 83_500_000
ROSTER_SIZE = 21
MIN_FORWARDS = 12
MIN_DEFENSEMEN = 6

MUST_INCLUDE = ["Auston Matthews", "William Nylander"]
MUST_EXCLUDE = ["Mitch Marner"]

EXCLUDE_LEAGUE_MIN = True
LEAGUE_MIN = 999_000

def to_group(pos):
    pos = str(pos).upper().strip()
    return "D" if pos in ["D", "LD", "RD"] else "F"

def main():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    df = con.execute(f"SELECT * FROM read_csv_auto('{S3_URL}')").df()

    df["cap_hit"] = pd.to_numeric(df["cap_hit"], errors="coerce")
    df["pred_mp_value"] = pd.to_numeric(df["pred_mp_value"], errors="coerce")
    df = df[df["cap_hit"].notna() & (df["cap_hit"] > 0)]
    df = df[df["pred_mp_value"].notna()]
    df["group"] = df["position"].apply(to_group)

    if EXCLUDE_LEAGUE_MIN:
        df = df[df["cap_hit"] > LEAGUE_MIN]

    if len(MUST_EXCLUDE) > 0:
        df = df[~df["Name"].isin(MUST_EXCLUDE)]

    nF = int((df["group"] == "F").sum())
    nD = int((df["group"] == "D").sum())
    f_min = min(MIN_FORWARDS, nF)
    d_min = min(MIN_DEFENSEMEN, nD)
    while f_min + d_min > ROSTER_SIZE and f_min > 0:
        f_min -= 1
    while f_min + d_min > ROSTER_SIZE and d_min > 0:
        d_min -= 1

    idx = df.index.tolist()
    F_idx = df.index[df["group"] == "F"].tolist()
    D_idx = df.index[df["group"] == "D"].tolist()

    model = cp_model.CpModel()
    x = {i: model.NewBoolVar(f"x_{i}") for i in idx}

    SCALE = 1000
    model.Maximize(sum(int(df.at[i, "pred_mp_value"] * SCALE) * x[i] for i in idx))
    model.Add(sum(int(df.at[i, "cap_hit"]) * x[i] for i in idx) <= CAP)
    model.Add(sum(x[i] for i in idx) == ROSTER_SIZE)
    model.Add(sum(x[i] for i in F_idx) >= f_min)
    model.Add(sum(x[i] for i in D_idx) >= d_min)

    for name in MUST_INCLUDE:
        rows = df.index[df["Name"] == name].tolist()
        for r in rows:
            model.Add(x[r] == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found with current constraints.")
        return

    chosen = [i for i in idx if solver.Value(x[i]) == 1]
    roster = df.loc[chosen, ["Name", "team", "position", "group", "cap_hit", "pred_mp_value", "mp_value"]].copy()

    roster["group"] = pd.Categorical(roster["group"], categories=["F", "D"], ordered=True)
    roster = roster.sort_values(["group", "pred_mp_value"], ascending=[True, False])

    roster.to_csv("optimized_roster.csv", index=False)
    print("Saved -> optimized_roster.csv")
    print("Total cap: ${:,.0f}".format(float(roster["cap_hit"].sum())))
    print("Total pred mp_value:", float(roster["pred_mp_value"].sum()))
    print("Counts:", roster["group"].value_counts().to_dict())

if __name__ == "__main__":
    main()
