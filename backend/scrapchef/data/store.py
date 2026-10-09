"""DATA -> INTELLIGENCE access functions. Intelligence code reads only through here."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2].parent
DB = ROOT / "data" / "warehouse.duckdb"
CFG = Path(__file__).resolve().parent.parent / "config"

GROUPS = ["SCRAP_3A", "SCRAP_16A_6A", "SCRAP_25A_3ACH", "IRON_UNITS", "OTHER_SCRAP"]


def cfg(name: str) -> dict:
    return yaml.safe_load((CFG / f"{name}.yaml").read_text())


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB), read_only=read_only)


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


def basket_profile() -> pd.DataFrame:
    return connect().execute("SELECT * FROM basket_profile").df()


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


def recipe_bounds() -> dict[str, tuple[float, float]]:
    """Historical per-group share of metallic charge [p5, p95] over the cohort (DERIVED)."""
    ht = heat_table()
    ht = ht[ht.in_cohort]
    out = {}
    for g in GROUPS:
        s = ht[g] / ht.total_charge_t
        out[g] = (float(s.quantile(0.05)), float(s.quantile(0.95)))
    return out


def code_split(days: int = 90) -> dict[str, list[tuple[int, str, float]]]:
    """Per group, the share of tonnes by real mill material code over the most recent `days`; groups with no recent
    usage fall back to all-time shares (DERIVED)."""
    con = connect()
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


def last_chem(n: int = 50) -> dict[str, float]:
    """Heel chemistry for the first planned heat: median of the last n measured heats (robust to outliers)."""
    row = connect().execute(f"""SELECT median(cu), median(sn), median(ni), median(mo) FROM
        (SELECT cu, sn, ni, mo FROM heat_chem WHERE cu IS NOT NULL ORDER BY sample_ts DESC LIMIT {n})""").fetchone()
    return dict(zip(["cu", "sn", "ni", "mo"], row))
