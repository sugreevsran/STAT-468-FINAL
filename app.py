import os
import logging
import duckdb
import pandas as pd
from dotenv import load_dotenv
from ortools.sat.python import cp_model
from shiny import App, ui, render, reactive
import plotly.express as px
from shinywidgets import output_widget, render_widget

load_dotenv()
S3_URL = os.getenv(
    "S3_URL",
    "https://stat468-final-project.s3.us-east-1.amazonaws.com/player_predictions.csv",
)

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

CAP = 83_500_000
ROSTER_SIZE = 21
MIN_FORWARDS = 12
MIN_DEFENSEMEN = 6
MUST_INCLUDE = ["Auston Matthews", "William Nylander"]
MUST_EXCLUDE = ["Mitch Marner"]
EXCLUDE_LEAGUE_MIN = True
LEAGUE_MIN = 999_000

def to_group(pos):
    p = str(pos).upper().strip()
    if p == "D" or p == "LD" or p == "RD":
        return "D"
    else:
        return "F"

def load_data():
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    df = con.execute("SELECT * FROM read_csv_auto($1)", [S3_URL]).df()
    return df

def optimize_roster(
    df,
    cap,
    roster_size,
    min_forwards,
    min_defense,
    must_include,
    must_exclude,
):
    df = df.copy()
    df["cap_hit"] = pd.to_numeric(df["cap_hit"], errors="coerce")
    df["pred_mp_value"] = pd.to_numeric(df["pred_mp_value"], errors="coerce")
    df = df[df["cap_hit"].notna() & (df["cap_hit"] > 0)]
    df = df[df["pred_mp_value"].notna()]
    df["group"] = df["position"].apply(to_group)
    if EXCLUDE_LEAGUE_MIN:
        df = df[df["cap_hit"] > LEAGUE_MIN]
    if must_exclude and len(must_exclude) > 0:
        df = df[~df["Name"].isin(must_exclude)]
    nF = int((df["group"] == "F").sum())
    nD = int((df["group"] == "D").sum())
    f_min = min(int(min_forwards), nF)
    d_min = min(int(min_defense), nD)
    while f_min + d_min > int(roster_size) and f_min > 0:
        f_min = f_min - 1
    while f_min + d_min > int(roster_size) and d_min > 0:
        d_min = d_min - 1
    idx = list(df.index)
    F_idx = []
    D_idx = []
    i = 0
    while i < len(idx):
        row_i = idx[i]
        if df.loc[row_i, "group"] == "F":
            F_idx.append(row_i)
        else:
            if df.loc[row_i, "group"] == "D":
                D_idx.append(row_i)
        i = i + 1
    model = cp_model.CpModel()
    x = {}
    i = 0
    while i < len(idx):
        row_i = idx[i]
        x[row_i] = model.NewBoolVar("x_" + str(row_i))
        i = i + 1
    SCALE = 1000
    obj_terms = []
    i = 0
    while i < len(idx):
        row_i = idx[i]
        val_scaled = int(df.loc[row_i, "pred_mp_value"] * SCALE)
        obj_terms.append(val_scaled * x[row_i])
        i = i + 1
    model.Maximize(sum(obj_terms))
    cap_terms = []
    i = 0
    while i < len(idx):
        row_i = idx[i]
        cap_terms.append(int(df.loc[row_i, "cap_hit"]) * x[row_i])
        i = i + 1
    model.Add(sum(cap_terms) <= int(cap))
    size_terms = []
    i = 0
    while i < len(idx):
        row_i = idx[i]
        size_terms.append(x[row_i])
        i = i + 1
    model.Add(sum(size_terms) == int(roster_size))
    f_terms = []
    i = 0
    while i < len(F_idx):
        row_i = F_idx[i]
        f_terms.append(x[row_i])
        i = i + 1
    model.Add(sum(f_terms) >= f_min)
    d_terms = []
    i = 0
    while i < len(D_idx):
        row_i = D_idx[i]
        d_terms.append(x[row_i])
        i = i + 1
    model.Add(sum(d_terms) >= d_min)
    if must_include and len(must_include) > 0:
        ni = 0
        while ni < len(must_include):
            name_needed = must_include[ni]
            rows = df.index[df["Name"] == name_needed].tolist()
            rj = 0
            while rj < len(rows):
                r = rows[rj]
                model.Add(x[r] == 1)
                rj = rj + 1
            ni = ni + 1
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    solver.Solve(model)
    chosen = []
    i = 0
    while i < len(idx):
        row_i = idx[i]
        if solver.Value(x[row_i]) == 1:
            chosen.append(row_i)
        i = i + 1
    cols = ["Name", "team", "position", "group", "cap_hit", "pred_mp_value"]
    if "mp_value" in df.columns:
        cols.append("mp_value")
    roster = df.loc[chosen, cols].copy()
    if not roster.empty:
        roster = roster.sort_values(["group", "pred_mp_value"], ascending=[True, False])
    total_cap = 0.0
    total_val = 0.0
    if not roster.empty:
        total_cap = float(roster["cap_hit"].sum())
        total_val = float(roster["pred_mp_value"].sum())
    return roster, total_cap, total_val

app_ui = ui.page_fluid(
    ui.h2("Simple Skater Roster Optimizer"),
    ui.input_numeric("cap", "Salary Cap ($)", CAP),
    ui.input_numeric("roster_size", "Roster Size", ROSTER_SIZE),
    ui.input_numeric("min_forwards", "Minimum Forwards", MIN_FORWARDS),
    ui.input_numeric("min_defense", "Minimum Defensemen", MIN_DEFENSEMEN),
    ui.input_text("must_include", "Must Include (comma separated)", ", ".join(MUST_INCLUDE)),
    ui.input_text("must_exclude", "Must Exclude (comma separated)", ", ".join(MUST_EXCLUDE)),
    ui.input_action_button("run", "Run Optimizer"),
    ui.hr(),
    ui.h4("Summary"),
    ui.output_text("summary"),
    ui.h4("Roster"),
    ui.output_data_frame("roster_table"),
    ui.h4("Cap vs Value"),
    output_widget("scatter"),
)

def server(input, output, session):
    @reactive.event(input.run)
    def run_optimizer():
        logging.info(
            "run clicked cap=%s roster=%s minF=%s minD=%s include=%s exclude=%s",
            input.cap(),
            input.roster_size(),
            input.min_forwards(),
            input.min_defense(),
            input.must_include(),
            input.must_exclude(),
        )
        df = load_data()
        must_inc_raw = input.must_include()
        must_exc_raw = input.must_exclude()
        must_inc = []
        if must_inc_raw is not None:
            parts = str(must_inc_raw).split(",")
            i = 0
            while i < len(parts):
                s = parts[i].strip()
                if s != "":
                    must_inc.append(s)
                i = i + 1
        must_exc = []
        if must_exc_raw is not None:
            parts = str(must_exc_raw).split(",")
            i = 0
            while i < len(parts):
                s = parts[i].strip()
                if s != "":
                    must_exc.append(s)
                i = i + 1
        return optimize_roster(
            df=df,
            cap=int(input.cap()),
            roster_size=int(input.roster_size()),
            min_forwards=int(input.min_forwards()),
            min_defense=int(input.min_defense()),
            must_include=must_inc,
            must_exclude=must_exc,
        )

    @output
    @render.text
    def summary():
        res = run_optimizer()
        if not res:
            return "Click Run Optimizer."
        roster, total_cap, total_val = res
        if roster.empty:
            return "No feasible roster."
        return f"Total Cap: ${total_cap:,.0f} | Total Value: {total_val:.2f}"

    @output
    @render.data_frame
    def roster_table():
        res = run_optimizer()
        if not res:
            return pd.DataFrame()
        roster, _, _ = res
        return roster.reset_index(drop=True)

    @output
    @render_widget
    def scatter():
        res = run_optimizer()
        if not res:
            return None
        roster, _, _ = res
        if roster.empty:
            return None
        fig = px.scatter(
            roster,
            x="cap_hit",
            y="pred_mp_value",
            color="group",
            hover_data=["Name", "team", "position"],
            labels={"cap_hit": "Cap Hit ($)", "pred_mp_value": "Predicted Value"},
            title="Cap Hit vs Predicted Value",
        )
        return fig

app = App(app_ui, server)
