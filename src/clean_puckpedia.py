import pandas as pd
import re

df = pd.read_excel("data/raw/puckpedia_raw.xlsx")

# we'll keep the important columns
df = df[["Name", "Pos", "GP", "Cap Hit", "Length", "Start Year"]]

def clean_name(n):
    if pd.isna(n):
        return ""
    n = re.sub(r"[^\x00-\x7F]+", "", str(n)).strip()
    if "," in n:
        last, first = n.split(",", 1)
        n = f"{first.strip()} {last.strip()}"
    return n

df["Name"] = df["Name"].apply(clean_name)

df.to_csv("data/processed/puckpedia_salaries.csv", index=False)
print(f"Saved cleaned salaries: {df.shape[0]} rows")
