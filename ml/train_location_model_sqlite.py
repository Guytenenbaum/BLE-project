import argparse
import sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def load_wide_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Build a wide table: one row per scan_id with 13 RSSI features as columns.
    Also includes true_loc and ts for ordering/analysis.
    """
    df = pd.read_sql_query(
        """
        SELECT s.scan_id, s.ts, s.true_loc, r.beacon_id, r.rssi
        FROM scans s
        JOIN raw_reads r ON r.scan_id = s.scan_id
        WHERE s.true_loc IS NOT NULL
        ORDER BY s.scan_id, r.beacon_id;
        """,
        conn,
    )

    wide = df.pivot_table(index=["scan_id", "ts", "true_loc"], columns="beacon_id", values="rssi")
    wide.columns = [f"rssi_b{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # Replace out-of-range marker (-200) with NaN so imputer can handle it
    rssi_cols = [c for c in wide.columns if c.startswith("rssi_b")]
    wide[rssi_cols] = wide[rssi_cols].replace(-200, np.nan)

    # Add simple “quality” features (very interview-friendly)
    wide["num_heard"] = wide[rssi_cols].notna().sum(axis=1)
    wide["max_rssi"] = wide[rssi_cols].max(axis=1, skipna=True)
    # wide["min_rssi"] = wide[rssi_cols].min(axis=1, skipna=True) #thats not -200
    wide["mean_rssi"] = wide[rssi_cols].mean(axis=1, skipna=True)

    return wide


def write_predictions(conn: sqlite3.Connection, preds_df: pd.DataFrame, model_version: str):
    conn.execute("DELETE FROM loc_predictions WHERE model_version = ?;", (model_version,))
    conn.executemany(
        """
        INSERT OR REPLACE INTO loc_predictions(scan_id, model_version, pred_loc, pred_conf)
        VALUES (?, ?, ?, ?);
        """,
        [
            (int(r.scan_id), model_version, str(r.pred_loc), float(r.pred_conf))
            for r in preds_df.itertuples(index=False)
        ],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to SQLite DB (data/wiliot.db)")
    parser.add_argument("--model-version", default="logreg_v1", help="String version tag")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    wide = load_wide_features(conn)

    feature_cols = [c for c in wide.columns if c.startswith("rssi_b")] + ["num_heard", "max_rssi", "mean_rssi"]
    X = wide[feature_cols].values
    y = wide["true_loc"].astype(str).values

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, wide["scan_id"].values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, n_jobs=None)),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate
    probs = model.predict_proba(X_test)
    pred = model.predict(X_test)
    conf = probs.max(axis=1)

    print("\n=== Classification report (test) ===")
    print(classification_report(y_test, pred))

    print("=== Confusion matrix (test) ===")
    print(confusion_matrix(y_test, pred))

    # Write predictions for ALL labeled scans (so events can run later)
    all_probs = model.predict_proba(X)
    all_pred = model.predict(X)
    all_conf = all_probs.max(axis=1)

    preds_df = pd.DataFrame({
        "scan_id": wide["scan_id"].values,
        "pred_loc": all_pred,
        "pred_conf": all_conf,
    })

    write_predictions(conn, preds_df, args.model_version)
    print(f"\nWrote {len(preds_df)} predictions to loc_predictions (model_version='{args.model_version}')")

    conn.close()


if __name__ == "__main__":
    main()
