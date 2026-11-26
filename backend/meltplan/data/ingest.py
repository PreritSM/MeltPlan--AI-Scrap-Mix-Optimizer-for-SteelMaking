"""DATA layer: raw CSV -> stg_* -> feature tables in DuckDB (`data/warehouse.duckdb`).

Run: `python -m meltplan.data.ingest`. Every cleaning rule is logged to `dq_issues`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "EAF Kaggle Dataset"
DB = ROOT / "data" / "warehouse.duckdb"
CFG = Path(__file__).resolve().parents[1] / "config"


# Builds an SQL expression that parses a decimal-comma text column as DOUBLE.
# Commas are replaced with dots before a TRY_CAST.
# Returns the SQL fragment as a string.
def N(c: str) -> str:
    return f"TRY_CAST(REPLACE({c}, ',', '.') AS DOUBLE)"


# Builds an SQL read_csv() call for a raw dataset file.
# All columns are read as text and malformed rows are tolerated.
# Returns the SQL fragment as a string.
def csv(f: str) -> str:
    return (f"read_csv('{RAW / f}', all_varchar=true, header=true, ignore_errors=true, "
            f"strict_mode=false, null_padding=true)")


# Builds the DuckDB warehouse from the raw Kaggle CSV files.
# Creates staging, grade map, heat, chemistry, energy and basket tables, logging DQ issues.
# Removes dropped-code heats, defines the analysis cohort and returns the connection.
def build(db: Path = DB, raw_dir: Path = RAW) -> duckdb.DuckDBPyConnection:
    mill = yaml.safe_load((CFG / "mill.yaml").read_text())
    gm = yaml.safe_load((CFG / "grade_map.yaml").read_text())
    db.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(db))
    for t in con.execute("SHOW TABLES").fetchall():
        con.execute(f"DROP TABLE IF EXISTS {t[0]}")
    con.execute("CREATE TABLE dq_issues(tbl VARCHAR, heatid VARCHAR, rule VARCHAR, action VARCHAR)")

    # material master (code -> group)
    rows = [(int(c), g, spec["category"]) for g, spec in gm["groups"].items() for c in spec["codes"]]
    rows += [(int(c), "NON_METALLIC", None) for c in gm["non_metallic"]]
    rows += [(int(c), "NON_METALLIC", None) for c in gm["dropped"]]  # their heats are removed below; this only avoids "unmapped" noise
    con.execute("CREATE TABLE grade_map(mat_code BIGINT, grade_group VARCHAR, category VARCHAR)")
    con.executemany("INSERT INTO grade_map VALUES (?,?,?)", rows)

    # stg_charge: typed basket charges (kg -> t). Negative amounts kept (corrections), netted later.
    con.execute(f"""
      CREATE TABLE stg_charge AS
      SELECT HEATID AS heatid, CAST(DATETIME AS TIMESTAMP) AS ts, CAST(MAT_CODE AS BIGINT) AS mat_code,
             MAT_DEC AS mat_name, {N('CHARGED_AMOUNT')}/1000.0 AS t
      FROM {csv('basket_charged.csv')} WHERE HEATID IS NOT NULL AND {N('CHARGED_AMOUNT')} IS NOT NULL""")
    con.execute("""INSERT INTO dq_issues SELECT 'basket_charged', heatid, 'negative_amount', 'netted into heat total'
                   FROM stg_charge WHERE t < 0""")

    # unmapped codes -> OTHER_SCRAP (logged)
    con.execute("""INSERT INTO dq_issues SELECT DISTINCT 'basket_charged', NULL, 'unmapped_code:' || mat_code, 'assigned OTHER_SCRAP'
                   FROM stg_charge WHERE mat_code NOT IN (SELECT mat_code FROM grade_map)""")
    con.execute("""
      CREATE TABLE heat_charge_group AS
      SELECT c.heatid, COALESCE(g.grade_group, 'OTHER_SCRAP') AS grade_group, GREATEST(sum(c.t), 0) AS t
      FROM stg_charge c LEFT JOIN grade_map g USING (mat_code)
      GROUP BY 1, 2 HAVING COALESCE(g.grade_group, 'OTHER_SCRAP') <> 'NON_METALLIC'""")

    # basket-level (distinct timestamps within heat = basket proxy; verified in profile report)
    con.execute("""
      CREATE TABLE heat_charge_basket AS
      SELECT heatid, dense_rank() OVER (PARTITION BY heatid ORDER BY ts) AS basket_no, ts, mat_code,
             any_value(mat_name) AS mat_name, sum(t) AS t
      FROM stg_charge GROUP BY heatid, ts, mat_code""")

    # heat_fact + chains
    con.execute(f"""
      CREATE TABLE heat_fact AS
      WITH h AS (
        SELECT g.heatid, g.total_charge_t, s.start_ts, s.last_charge_ts
        FROM (SELECT heatid, sum(t) AS total_charge_t FROM heat_charge_group GROUP BY 1) g
        JOIN (SELECT heatid, min(ts) AS start_ts, max(ts) AS last_charge_ts FROM stg_charge GROUP BY 1) s USING (heatid)),
      nb AS (SELECT heatid, count(DISTINCT ts) AS n_baskets FROM stg_charge GROUP BY 1),
      o AS (SELECT *, date_diff('minute', lag(start_ts) OVER (ORDER BY start_ts), start_ts) AS gap_min FROM h)
      SELECT o.*, nb.n_baskets, row_number() OVER (ORDER BY start_ts) AS seq_no,
             sum(CASE WHEN gap_min IS NULL OR gap_min > {mill['heel_chain_break_h'] * 60} THEN 1 ELSE 0 END)
               OVER (ORDER BY start_ts) AS chain_id,
             (gap_min IS NULL OR gap_min > {mill['heel_chain_break_h'] * 60}) AS chain_start
      FROM o JOIN nb USING (heatid) ORDER BY start_ts""")
    con.execute("""INSERT INTO dq_issues SELECT 'heat_fact', heatid, 'chain_start', 'heel chem = trailing median'
                   FROM heat_fact WHERE chain_start""")

    # heat_chem: single EAF-final sample per heat (POSITIONROW = 1 for all in profile); values are wt %
    els = ["C", "SI", "MN", "P", "S", "CU", "CR", "MO", "NI", "AS", "SN", "N"]
    sel = ", ".join(f"{N('VAL' + e)} AS {e.lower()}" for e in els)
    con.execute(f"""
      CREATE TABLE heat_chem AS
      SELECT HEATID AS heatid, CAST(DATETIME AS TIMESTAMP) AS sample_ts, {sel}
      FROM {csv('eaf_final_chemical_measurements.csv')}
      QUALIFY row_number() OVER (PARTITION BY HEATID ORDER BY CAST(POSITIONROW AS INT) DESC) = 1""")
    con.execute("""INSERT INTO dq_issues SELECT 'heat_chem', heatid, 'null_cu', 'excluded from fitting'
                   FROM heat_chem WHERE cu IS NULL""")

    # heat_energy. DURATION is 'mm: ss' (verified against heat 6F0020). Two candidates stored:
    #   mwh_integrated = sum(MW * duration_h);  mwh_sum_mw = sum(MW)  -> which is right is Open (see report).
    con.execute(f"""
      CREATE TABLE tr AS
      SELECT HEATID AS heatid, {N('MW')} AS mw,
             CAST(split_part(trim(DURATION), ':', 1) AS INT) * 60 + CAST(trim(split_part(DURATION, ':', 2)) AS INT) AS dur_s
      FROM {csv('eaf_transformer.csv')} WHERE DURATION LIKE '%:%'""")
    con.execute(f"""
      CREATE TABLE heat_energy AS
      WITH t AS (SELECT heatid, sum(mw * dur_s / 3600.0) AS mwh_integrated, sum(mw) AS mwh_sum_mw,
                        sum(dur_s) / 60.0 AS power_on_min FROM tr GROUP BY 1),
           o AS (SELECT HEATID AS heatid, max({N('O2_AMOUNT')}) AS o2_nm3, max({N('GAS_AMOUNT')}) AS gas_nm3
                 FROM {csv('eaf_gaslance_mat.csv')} GROUP BY 1),
           c AS (SELECT HEATID AS heatid, max({N('INJ_AMOUNT_CARBON')}) / 1000.0 AS inj_c_t
                 FROM {csv('inj_mat.csv')} GROUP BY 1)
      SELECT * FROM t LEFT JOIN o USING (heatid) LEFT JOIN c USING (heatid)""")
    con.execute("DROP TABLE tr")

    # heat_additions: non-charge additions to EAF (t), for the mass balance
    con.execute(f"""
      CREATE TABLE heat_additions AS
      SELECT HEATID AS heatid, CAST(MAT_CODE AS BIGINT) AS mat_code, MAT_DEC AS mat_name,
             sum({N('CHARGE_AMOUNT')}) / 1000.0 AS t
      FROM {csv('eaf_added_materials.csv')} GROUP BY 1, 2, 3""")

    # basket_profile: mass per basket position (positive-mass baskets only)
    con.execute("""
      CREATE TABLE basket_profile AS
      WITH b AS (SELECT heatid, basket_no, sum(t) AS t FROM heat_charge_basket GROUP BY 1, 2)
      SELECT basket_no, count(*) AS n, quantile_cont(t, 0.1) AS p10, quantile_cont(t, 0.5) AS p50,
             quantile_cont(t, 0.9) AS p90 FROM b WHERE t > 5 AND basket_no <= 3 GROUP BY 1 ORDER BY 1""")

    # DEMO SIMPLIFICATION: remove every heat that charged a dropped code; the first surviving heat after each one becomes a chain start
    dropped = ",".join(str(int(c)) for c in gm["dropped"])
    con.execute(f"CREATE TABLE dropped_heats AS SELECT DISTINCT heatid FROM stg_charge WHERE mat_code IN ({dropped})")
    con.execute("""
      CREATE TABLE succ AS
      SELECT heatid FROM (SELECT heatid, chain_start, lag(heatid) OVER (ORDER BY seq_no) AS prev FROM heat_fact)
      WHERE NOT chain_start AND prev IN (SELECT heatid FROM dropped_heats) AND heatid NOT IN (SELECT heatid FROM dropped_heats)""")
    con.execute("""INSERT INTO dq_issues SELECT 'heat_fact', heatid, 'dropped_scrap_code', 'heat removed (demo simplification)' FROM dropped_heats""")
    con.execute("""INSERT INTO dq_issues SELECT 'heat_fact', heatid, 'chain_start', 'predecessor removed; heel chem = trailing median' FROM succ""")
    con.execute("UPDATE heat_fact SET chain_start = TRUE WHERE heatid IN (SELECT heatid FROM succ)")
    for t in ["stg_charge", "heat_charge_group", "heat_charge_basket", "heat_fact", "heat_chem", "heat_energy", "heat_additions"]:
        con.execute(f"DELETE FROM {t} WHERE heatid IN (SELECT heatid FROM dropped_heats)")
    con.execute("""CREATE OR REPLACE TABLE heat_fact AS
                   SELECT * EXCLUDE (chain_id), sum(CASE WHEN chain_start THEN 1 ELSE 0 END) OVER (ORDER BY seq_no) AS chain_id
                   FROM heat_fact ORDER BY seq_no""")
    con.execute("DROP TABLE succ"); con.execute("DROP TABLE dropped_heats")

    # analysis cohort
    con.execute("""
      CREATE TABLE cohort AS
      SELECT f.heatid FROM heat_fact f JOIN heat_chem c USING (heatid) JOIN heat_energy e USING (heatid)
      WHERE c.cu IS NOT NULL AND f.total_charge_t BETWEEN
        (SELECT quantile_cont(total_charge_t, 0.005) FROM heat_fact) AND
        (SELECT quantile_cont(total_charge_t, 0.995) FROM heat_fact)""")
    con.execute("""INSERT INTO dq_issues SELECT 'heat_fact', heatid, 'not_in_cohort', 'missing chem/energy or charge outside p0.5-p99.5'
                   FROM heat_fact WHERE heatid NOT IN (SELECT heatid FROM cohort)""")
    return con


# Runs the ingestion pipeline from the command line.
# Prints the row count of each warehouse table.
# Also prints a summary of data-quality issues by rule.
def main() -> None:
    con = build()
    for t in ["stg_charge", "heat_fact", "heat_charge_group", "heat_charge_basket", "heat_chem",
              "heat_energy", "heat_additions", "basket_profile", "cohort"]:
        print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
    print(con.execute("SELECT rule, count(*) FROM dq_issues GROUP BY 1 ORDER BY 2 DESC").fetchall())


if __name__ == "__main__":
    sys.exit(main())
