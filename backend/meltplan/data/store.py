"""DATA -> INTELLIGENCE access functions. Intelligence code reads only through here."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2].parent
DB = ROOT / "data" / "warehouse.duckdb"
CFG = Path(__file__).resolve().parent.parent / "config"

GROUPS = ["SCRAP_3A", "SCRAP_16A_6A", "SCRAP_25A_3ACH", "IRON_UNITS"]


# Loads a YAML config file from the package config directory by name.
# For example cfg('mill') reads config/mill.yaml.
# Returns the parsed contents as a dict.
def cfg(name: str) -> dict:
    return yaml.safe_load((CFG / f"{name}.yaml").read_text())


# Opens a DuckDB connection to the data warehouse file.
# Connections are read-only by default so intelligence code cannot write.
# Returns the DuckDB connection object.
def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB), read_only=read_only)


# Builds one row per heat with chemistry, chain info and cohort flag.
# Pivots charged tonnes by grade group into columns and joins them on.
# Returns the time-ordered DataFrame used by estimators and back-tests.
def heat_table() -> pd.DataFrame:
    """One row per heat (all heats, time ordered) with group tonnes, chem (wt %), chain info, cohort flag."""
    con = connect()
    f = con.execute("""
      SELECT f.heatid, f.seq_no, f.chain_id, f.chain_start, f.start_ts, f.total_charge_t, f.n_baskets,
             c.cu, c.sn, c.ni, c.mo, c.cr, (f.heatid IN (SELECT heatid FROM cohort)) AS in_cohort
      FROM heat_fact f LEFT JOIN heat_chem c USING (heatid) ORDER BY f.seq_no""").df()
    g = con.execute("SELECT heatid, grade_group, t FROM heat_charge_group").df()
    w = g.pivot_table(index="heatid", columns="grade_group", values="t", aggfunc="sum").reindex(columns=GROUPS).fillna(0.0)
    return f.merge(w, left_on="heatid", right_index=True, how="left").fillna({k: 0.0 for k in GROUPS})


# Loads the basket profile table from the warehouse.
# It holds typical basket masses per basket number.
# Returns the table as a DataFrame.
def basket_profile() -> pd.DataFrame:
    return connect().execute("SELECT * FROM basket_profile").df()


# Returns prior mean and SD of an element's content for each grade group.
# Values come from the grade map config (published typical values, ASSUMED).
# For non-Cu elements the SD is taken as half the mean.
def priors(element: str) -> tuple[list[float], list[float]]:
    """(mean, sd) wt % per group from Table H2 via grade_map (ASSUMED)."""
    gm, cats = cfg("grade_map"), cfg("grade_map")["categories"]
    mu, sd = [], []
    for g in GROUPS:
        c = cats[gm["groups"][g]["category"]]
        if element == "cu":
            m, s = c["cu"]
        else:
            m, s = c[element], 0.5 * c[element]
        mu.append(m)
        sd.append(s)
    return mu, sd


# Returns the assumed metallic yield for each grade group.
# Read from the grade map config in GROUPS order.
# The single source for tap weight in the estimator, optimiser and replay.
def group_yields() -> np.ndarray:
    """Metallic yield per group (ASSUMED, Table H2 via grade_map). The single source for tap weight: estimator, optimiser, replay."""
    gm = cfg("grade_map")
    return np.array([gm["categories"][gm["groups"][g]["category"]]["yield"] for g in GROUPS], float)


# Derives historical recipe bounds for each grade group.
# Computes each group's share of total charge over the cohort heats.
# Returns the 5th and 95th percentile share per group.
def recipe_bounds() -> dict[str, tuple[float, float]]:
    """Historical per-group share of metallic charge [p5, p95] over the cohort (DERIVED)."""
    ht = heat_table()
    ht = ht[ht.in_cohort]
    out = {}
    for g in GROUPS:
        s = ht[g] / ht.total_charge_t
        out[g] = (float(s.quantile(0.05)), float(s.quantile(0.95)))
    return out


# Computes how each grade group splits across real mill material codes.
# Uses tonnage shares over the most recent `days`, falling back to all-time shares.
# Returns, per group, a list of (mat_code, mat_name, share) tuples.
def code_split(days: int = 90) -> dict[str, list[tuple[int, str, float]]]:
    """Per group, the share of tonnes by real mill material code over the most recent `days`; groups with no recent
    usage fall back to all-time shares (DERIVED)."""
    con = connect()
    # Queries tonnage per material code within each grade group.
    # An extra SQL WHERE fragment restricts the time window.
    # Returns each group's codes with their share of the group's tonnes.
    def shares(where: str):
        df = con.execute(f"""
          WITH m AS (SELECT max(ts) m FROM stg_charge)
          SELECT g.grade_group, c.mat_code, any_value(c.mat_name) AS mat_name, sum(c.t) AS t
          FROM stg_charge c JOIN grade_map g USING (mat_code), m
          WHERE g.grade_group <> 'NON_METALLIC' AND c.t > 0 {where}
          GROUP BY 1, 2""").df()
        out = {}
        for g, sub in df.groupby("grade_group"):
            tot = sub.t.sum()
            out[g] = [(int(r.mat_code), r.mat_name, float(r.t / tot)) for r in sub.itertuples()]
        return out
    allt = shares("")
    recent = shares(f"AND c.ts > m.m - INTERVAL {days} DAY")
    return {g: recent.get(g) or allt[g] for g in allt}


# Estimates the heel chemistry for the first planned heat.
# Takes the median of the last n measured heats for Cu, Sn, Ni and Mo.
# Returns a dict keyed by element.
def last_chem(n: int = 50) -> dict[str, float]:
    """Heel chemistry for the first planned heat: median of the last n measured heats (robust to outliers)."""
    row = connect().execute(f"""SELECT median(cu), median(sn), median(ni), median(mo) FROM
        (SELECT cu, sn, ni, mo FROM heat_chem WHERE cu IS NOT NULL ORDER BY sample_ts DESC LIMIT {n})""").fetchone()
    return dict(zip(["cu", "sn", "ni", "mo"], row))
