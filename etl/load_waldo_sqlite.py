# etl/load_waldo_sqlite.py
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def run_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    """
    Creates all SQL tables in the SQLite database
    :param conn:
    :param schema_path:
    :return:
    """
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()


def get_columns(df: pd.DataFrame):
    loc_col = "location"
    ts_col = "date"
    rssi_cols = [f"b{3000+i}" for i in range(1, 14)]  # b3001..b3013

    missing = [c for c in [loc_col, ts_col, *rssi_cols] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV format mismatch. Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return loc_col, ts_col, rssi_cols




def parse_timestamp_series(s: pd.Series) -> pd.Series:
    """
    Convert various timestamp formats to ISO strings.
    If parsing fails, create synthetic timestamps.
    """
    # if already in datetime format
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Try numeric epoch seconds
    if pd.api.types.is_numeric_dtype(s):
        # Heuristic: if values are huge, might be ms
        v = s.dropna().astype(float)
        if len(v) > 0 and v.median() > 1e12:
            dt = pd.to_datetime(s, unit="ms", errors="coerce")
        else:
            dt = pd.to_datetime(s, unit="s", errors="coerce")
        if dt.notna().mean() > 0.7:
            return dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Try general string parsing
    dt = pd.to_datetime(s.astype(str), errors="coerce", infer_datetime_format=True)
    if dt.notna().mean() > 0.7:
        return dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # if all the above didnt work we'll go for synthetic timestamps
    base = datetime(2026, 1, 1, 0, 0, 0)
    return pd.Series([(base + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(len(s))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labeled CSV (Waldo).")
    parser.add_argument("--db", required=True, help="Path to SQLite DB file to create/use.")
    parser.add_argument("--schema", required=True, help="Path to sql/schema.sql")
    parser.add_argument("--tag-id", default="tag_0", help="Tag ID to assign (dataset is often a single trajectory).")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing tables before loading.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)
    schema_path = Path(args.schema)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Guess columns
    loc_col, ts_col, rssi_cols = get_columns(df)

    # Normalize timestamps
    df[ts_col] = parse_timestamp_series(df[ts_col])

    # Normalize location
    df[loc_col] = df[loc_col].astype(str)

    # Keep only needed
    use_df = df[[loc_col, ts_col] + rssi_cols].copy()

    # Connect DB
    conn = sqlite3.connect(db_path.as_posix())
    conn.execute("PRAGMA foreign_keys = ON;")

    if args.drop_existing:
        conn.executescript(
            """
            DROP TABLE IF EXISTS events;
            DROP TABLE IF EXISTS loc_predictions;
            DROP TABLE IF EXISTS raw_reads;
            DROP TABLE IF EXISTS scans;
            """
        )
        conn.commit()

    # Create tables
    run_schema(conn, schema_path)

    # Insert scans
    scans_rows = []
    for i, row in use_df.iterrows():
        scan_id = i + 1
        scans_rows.append((scan_id, args.tag_id, row[ts_col], row[loc_col], "train"))  # default split=train for now

    conn.executemany(
        "INSERT OR REPLACE INTO scans(scan_id, tag_id, ts, true_loc, split) VALUES (?, ?, ?, ?, ?);",
        scans_rows,
    )

    # Insert raw_reads (long format)
    raw_rows = []
    # Map rssi cols -> beacon_id 1..13
    for i, row in use_df.iterrows():
        scan_id = i + 1
        for b_idx, col in enumerate(rssi_cols, start=1):
            rssi_val = float(row[col])
            raw_rows.append((scan_id, b_idx, rssi_val))

    conn.executemany(
        "INSERT OR REPLACE INTO raw_reads(scan_id, beacon_id, rssi) VALUES (?, ?, ?);",
        raw_rows,
    )

    conn.commit()

    # Quick sanity checks
    n_scans = conn.execute("SELECT COUNT(*) FROM scans;").fetchone()[0]
    n_reads = conn.execute("SELECT COUNT(*) FROM raw_reads;").fetchone()[0]
    n_locs = conn.execute("SELECT COUNT(DISTINCT true_loc) FROM scans WHERE true_loc IS NOT NULL;").fetchone()[0]

    print("Loaded successfully ✅")
    print(f"DB: {db_path}")
    print(f"scans: {n_scans}")
    print(f"raw_reads: {n_reads} (expected scans*13 = {n_scans*13})")
    print(f"unique labeled locations: {n_locs}")
    print(f"Guessed columns: loc={loc_col}, ts={ts_col}, rssi_cols={rssi_cols}")

    conn.close()


if __name__ == "__main__":
    main()
