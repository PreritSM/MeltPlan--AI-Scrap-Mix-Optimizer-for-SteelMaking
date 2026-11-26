"""Phase 0 data profiling: `python -m meltplan.data.profile` -> reports/profile.md.

Schema-agnostic first pass over the Kaggle EAF CSVs. All columns are read as text
(decimal commas are common in this dataset) and cast explicitly where needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "EAF Kaggle Dataset"
REPORT = ROOT / "reports" / "profile.md"

# Files read in full for per-column profiling; the two huge process-signal files
# are profiled by row count / heat coverage only.
BIG = {"eaf_gaslance_mat.csv", "inj_mat.csv"}


# Builds an SQL expression casting a decimal-comma text column to DOUBLE.
# Commas are replaced with dots before a TRY_CAST.
# Returns the SQL fragment as a string.
def num(col: str) -> str:
    """SQL expression casting a text column with decimal comma to DOUBLE."""
    return f"TRY_CAST(REPLACE({col}, ',', '.') AS DOUBLE)"


# Registers a raw CSV file as a DuckDB view.
# The view is named after the file stem and reads every column as text.
# Returns the view name.
def table(con: duckdb.DuckDBPyConnection, f: Path) -> str:
    name = f.stem
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv('{f}', all_varchar=true, header=true, ignore_errors=true, strict_mode=false, null_padding=true)"
    )
    return name


# Formats rows and column names as a Markdown table.
# None values are rendered as empty cells.
# Returns the table as a string.
def md_table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


# Runs an SQL query on a connection.
# Collects all result rows and the column names.
# Returns (rows, column names).
def q(con, sql):
    cur = con.execute(sql)
    return cur.fetchall(), [d[0] for d in cur.description]


# Profiles the raw Kaggle EAF CSV files.
# Covers files, join coverage, charges, codes, baskets, chemistry, energy and time gaps.
# Writes the result to reports/profile.md.
def main() -> None:
    con = duckdb.connect()
    files = sorted(RAW.glob("*.csv"))
    out = ["# Data profile (auto-generated)\n", f"Source: `{RAW.name}/`\n"]
    tabs = {f.stem: table(con, f) for f in files}

    # --- per-file overview
    out.append("## 1. Files\n")
    rows = []
    for name in tabs:
        cols = [c[0] for c in con.execute(f"DESCRIBE {name}").fetchall()]
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        h = (
            con.execute(f"SELECT count(DISTINCT HEATID) FROM {name}").fetchone()[0]
            if "HEATID" in cols
            else None
        )
        rows.append((name, n, h, ", ".join(cols)))
    out.append(md_table(rows, ["file", "rows", "distinct HEATID", "columns"]) + "\n")

    # --- join coverage
    out.append("## 2. Join coverage (distinct HEATIDs)\n")
    hn = [n for n in tabs if "HEATID" in [c[0] for c in con.execute(f"DESCRIBE {n}").fetchall()]]
    for n in hn:
        con.execute(f"CREATE OR REPLACE TEMP TABLE h_{n} AS SELECT DISTINCT HEATID FROM {n}")
    rows = []
    for a in hn:
        rows.append(
            (a, *[
                con.execute(f"SELECT count(*) FROM h_{a} JOIN h_{b} USING (HEATID)").fetchone()[0]
                for b in hn
            ])
        )
    out.append(md_table(rows, ["file \\ ∩", *hn]) + "\n")
    cohort = con.execute(
        "SELECT count(*) FROM h_basket_charged JOIN h_eaf_final_chemical_measurements USING (HEATID) "
        "JOIN h_eaf_transformer USING (HEATID)"
    ).fetchone()[0]
    out.append(f"**Analysis cohort** (basket ∩ final chem ∩ transformer): **{cohort}** heats\n")

    # --- charge amounts / units
    out.append("## 3. Charged amounts (units)\n")
    r, c = q(con, f"""
      WITH h AS (SELECT HEATID, sum({num('CHARGED_AMOUNT')}) AS tot, count(DISTINCT DATETIME) AS n_charges
                 FROM basket_charged GROUP BY 1)
      SELECT count(*) heats, quantile_cont(tot,0.005) p0_5, quantile_cont(tot,0.5) p50,
             quantile_cont(tot,0.995) p99_5, quantile_cont(n_charges,0.5) median_baskets FROM h""")
    out.append(md_table(r, c) + "\n\n(If p50 ≈ 130000–150000 the unit is kg; plan expects 130–150 t per heat.)\n")

    out.append("## 4. Material codes in basket_charged (Pareto)\n")
    r, c = q(con, f"""
      WITH t AS (SELECT MAT_CODE, any_value(MAT_DEC) AS mat_name, sum({num('CHARGED_AMOUNT')}) kg,
                        count(DISTINCT HEATID) heats FROM basket_charged GROUP BY 1)
      SELECT MAT_CODE, mat_name, round(kg/1000) tonnes, heats,
             round(100*kg/sum(kg) OVER (),2) pct,
             round(100*sum(kg) OVER (ORDER BY kg DESC)/sum(kg) OVER (),1) cum_pct
      FROM t ORDER BY kg DESC LIMIT 40""")
    out.append(md_table(r, c) + "\n")
    n_codes = con.execute("SELECT count(DISTINCT MAT_CODE) FROM basket_charged").fetchone()[0]
    out.append(f"\nDistinct codes: {n_codes}. Names per code >1: "
               f"{con.execute('SELECT count(*) FROM (SELECT MAT_CODE FROM basket_charged GROUP BY 1 HAVING count(DISTINCT MAT_DEC)>1)').fetchone()[0]}\n")

    out.append("## 5. Baskets\n")
    r, c = q(con, f"""
      WITH b AS (SELECT HEATID, DATETIME, sum({num('CHARGED_AMOUNT')}) kg FROM basket_charged GROUP BY 1,2),
           n AS (SELECT HEATID, DATETIME, row_number() OVER (PARTITION BY HEATID ORDER BY DATETIME) pos, kg FROM b)
      SELECT pos, count(*) n, quantile_cont(kg,0.1) p10, quantile_cont(kg,0.5) p50, quantile_cont(kg,0.9) p90
      FROM n WHERE pos<=6 GROUP BY 1 ORDER BY 1""")
    out.append("Distinct DATETIME per heat treated as a basket (to be verified):\n\n" + md_table(r, c) + "\n")

    out.append("## 6. Final chemistry (EAF)\n")
    r, c = q(con, "SELECT POSITIONROW, count(*) n FROM eaf_final_chemical_measurements GROUP BY 1 ORDER BY 1")
    out.append("Samples per row index:\n\n" + md_table(r, c) + "\n")
    rows = []
    for el in ["CU", "SN", "NI", "MO", "CR", "C", "SI", "MN", "P", "S"]:
        r, _ = q(con, f"""SELECT count({num('VAL'+el)}), round(quantile_cont({num('VAL'+el)},0.01),4),
                          round(quantile_cont({num('VAL'+el)},0.5),4), round(quantile_cont({num('VAL'+el)},0.99),4)
                          FROM eaf_final_chemical_measurements""")
        rows.append((el, *r[0]))
    out.append("\n" + md_table(rows, ["element", "n", "p1", "p50", "p99"]) + "\n")

    out.append("## 7. Transformer / energy\n")
    r, c = q(con, "SELECT DURATION, MW, TAP FROM eaf_transformer LIMIT 8")
    out.append(md_table(r, c) + "\n\nDURATION is text like ' 10: 40' (mm:ss or hh:mm — verify).\n")

    out.append("## 8. Time ordering and heat gaps\n")
    r, c = q(con, f"""
      WITH s AS (SELECT HEATID, min(CAST(DATETIME AS TIMESTAMP)) t0 FROM basket_charged GROUP BY 1),
           g AS (SELECT t0, date_diff('minute', lag(t0) OVER (ORDER BY t0), t0) gap FROM s)
      SELECT min(t0) first_heat, max(t0) last_heat, count(*) heats,
             quantile_cont(gap,0.5) gap_p50_min, quantile_cont(gap,0.99) gap_p99_min,
             sum(CASE WHEN gap>1440 THEN 1 ELSE 0 END) gaps_over_24h,
             round(count(*)/ (date_diff('day', min(t0), max(t0))+1.0),2) heats_per_day FROM g""")
    out.append(md_table(r, c) + "\n")

    out.append("## 9. Tap-weight search\n")
    hits = []
    for name in tabs:
        for col, *_ in con.execute(f"DESCRIBE {name}").fetchall():
            if any(k in col.upper() for k in ("TAP", "WEIGHT", "WGT", "MASS", "NETTO", "NET")):
                hits.append((name, col))
    out.append(md_table(hits, ["file", "candidate column"]) + "\n")

    out.append("## 10. ferro.csv vs scrap codes\n")
    r, c = q(con, 'SELECT MAT_CODE, Description, Cu, Sn, Ni, Mo FROM ferro ORDER BY 1')
    out.append(md_table(r, c) + "\n")
    ov = con.execute(
        "SELECT count(DISTINCT f.MAT_CODE) FROM ferro f JOIN basket_charged b USING (MAT_CODE)"
    ).fetchone()[0]
    out.append(f"\nferro codes appearing in basket_charged: **{ov}**\n")

    out.append("## 11. eaf_added_materials top codes\n")
    r, c = q(con, f"SELECT MAT_CODE, any_value(MAT_DEC) AS mat_name, count(*) n, round(sum({num('CHARGE_AMOUNT')})/1000,1) tonnes "
                  "FROM eaf_added_materials GROUP BY 1 ORDER BY 4 DESC LIMIT 15")
    out.append(md_table(r, c) + "\n")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(out))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
