import argparse
import sqlite3
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Grid distance (for "loss")
# -----------------------------
def loc_to_xy(loc: str) -> Optional[Tuple[int, int]]:
    if loc is None:
        return None
    s = str(loc).strip().upper()
    import re
    m = re.match(r"^([A-Z]+)\s*(\d+)$", s)
    if not m:
        return None
    letters = m.group(1)
    y = int(m.group(2))
    x = 0
    for ch in letters:
        x = x * 26 + (ord(ch) - ord("A") + 1)
    return x, y


def manhattan_distance_loc(a: str, b: str) -> Optional[int]:
    pa = loc_to_xy(a)
    pb = loc_to_xy(b)
    if pa is None or pb is None:
        return None
    return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])


def mean_manhattan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dists = []
    for t, p in zip(y_true, y_pred):
        d = manhattan_distance_loc(t, p)
        if d is not None:
            dists.append(d)
    return float(np.mean(dists)) if dists else float("nan")


def within_k_steps(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    ok = 0
    tot = 0
    for t, p in zip(y_true, y_pred):
        d = manhattan_distance_loc(t, p)
        if d is None:
            continue
        tot += 1
        if d <= k:
            ok += 1
    return (ok / tot) if tot else float("nan")


# -----------------------------
# Features
# -----------------------------
def load_wide_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Build wide features for ALL scans (labeled + unlabeled).
    true_loc remains nullable for unlabeled scans.
    """
    df = pd.read_sql_query(
        """
        SELECT s.scan_id, s.ts, s.true_loc, r.beacon_id, r.rssi
        FROM scans s
        JOIN raw_reads r ON r.scan_id = s.scan_id
        ORDER BY s.scan_id, r.beacon_id;
        """,
        conn,
    )

    wide = df.pivot_table(
        index=["scan_id", "ts", "true_loc"],
        columns="beacon_id",
        values="rssi",
        aggfunc="mean",
    )
    wide.columns = [f"rssi_b{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    rssi_cols = [c for c in wide.columns if c.startswith("rssi_b")]
    wide[rssi_cols] = wide[rssi_cols].replace(-200, np.nan)

    wide["num_heard"] = wide[rssi_cols].notna().sum(axis=1)
    wide["max_rssi"] = wide[rssi_cols].max(axis=1, skipna=True)
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


def make_knn_model(n_neighbors: int, weights: str) -> Pipeline:
    # scaler matters for kNN because distances are scale-sensitive
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
        ]
    )


def parse_int_grid(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Empty neighbors grid. Example: --neighbors-grid 3,5,7,9,11")
    return vals


# -----------------------------
# Merge sparse labels
# -----------------------------
def build_sparse_merge_map(labels: np.ndarray, min_count: int) -> Dict[str, str]:
    """
    Build mapping from sparse labels -> nearest dense label (by Manhattan distance on grid).

    - Dense = count >= min_count
    - Sparse = count < min_count

    Any label that can't be parsed is merged to 'OTHER'.
    If there are no dense labels (rare), merges everything to 'OTHER'.
    """
    counts = pd.Series(labels).value_counts()
    dense = counts[counts >= min_count].index.tolist()
    sparse = counts[counts < min_count].index.tolist()

    mapping: Dict[str, str] = {}

    if not dense:
        for s in sparse:
            mapping[str(s)] = "OTHER"
        return mapping

    # Precompute coords for dense labels that are parseable
    dense_coords = []
    for d in dense:
        xy = loc_to_xy(str(d))
        if xy is not None:
            dense_coords.append((str(d), xy))

    for s in sparse:
        s = str(s)
        s_xy = loc_to_xy(s)
        if s_xy is None or not dense_coords:
            mapping[s] = "OTHER"
            continue

        best_d = None
        best_dist = None
        for d, (dx, dy) in dense_coords:
            dist = abs(dx - s_xy[0]) + abs(dy - s_xy[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_d = d

        mapping[s] = best_d if best_d is not None else "OTHER"

    return mapping


def apply_label_mapping(y: np.ndarray, mapping: Dict[str, str]) -> np.ndarray:
    if not mapping:
        return y
    return np.array([mapping.get(str(lbl), str(lbl)) for lbl in y], dtype=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to SQLite DB (data/wiliot.db)")
    parser.add_argument("--model-version", default="knn_kfold_v2_merged", help="Version tag written to loc_predictions")
    parser.add_argument("--kfold", type=int, default=5, help="Number of CV folds (StratifiedKFold)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--neighbors-grid",
        default="3,5,7,9,11,15",
        help="Comma-separated list of k values to try (e.g. 3,5,7,9,11).",
    )
    parser.add_argument(
        "--weights",
        default="distance",
        choices=["uniform", "distance"],
        help="kNN voting weights.",
    )

    # NEW: merge sparse areas
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=5,
        help="Any location with fewer than this many labeled samples is merged into nearest dense location.",
    )

    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    wide_all = load_wide_features(conn)
    feature_cols = [c for c in wide_all.columns if c.startswith("rssi_b")] + ["num_heard", "max_rssi", "mean_rssi"]

    labeled = wide_all[wide_all["true_loc"].notna()].copy()
    if labeled.empty:
        raise RuntimeError("No labeled scans found (true_loc is NULL everywhere).")

    X_lab = labeled[feature_cols].values
    y_lab_raw = labeled["true_loc"].astype(str).values

    # Merge sparse labels BEFORE CV + training
    merge_map = build_sparse_merge_map(y_lab_raw, min_count=args.min_class_count)
    y_lab = apply_label_mapping(y_lab_raw, merge_map)

    # Print merge summary
    if merge_map:
        counts_before = pd.Series(y_lab_raw).value_counts()
        counts_after = pd.Series(y_lab).value_counts()
        n_sparse = sum(counts_before < args.min_class_count)
        print(f"\n=== Sparse merge enabled ===")
        print(f"min_class_count={args.min_class_count}")
        print(f"Sparse classes merged: {n_sparse}")
        print(f"Classes before: {counts_before.shape[0]} | after: {counts_after.shape[0]}")
        # show a few examples
        ex = list(merge_map.items())[:12]
        if ex:
            print("Example merges (sparse -> merged_to):")
            for a, b in ex:
                print(f"  {a} -> {b}")
    else:
        print("\n=== Sparse merge disabled (no sparse classes found) ===")

    # If after merging we still have some classes smaller than kfold, you should reduce kfold.
    # We'll do a safe automatic clamp here (so it never crashes):
    min_class_after = int(pd.Series(y_lab).value_counts().min())
    if args.kfold > min_class_after:
        print(f"\n[Note] Reducing kfold from {args.kfold} to {min_class_after} because smallest class has {min_class_after} samples.")
        args.kfold = max(2, min_class_after)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    neighbors_list = parse_int_grid(args.neighbors_grid)

    best_k = None
    best_score = (float("inf"), -float("inf"))  # (mean_manhattan, macro_f1)

    print(f"\n=== kNN baseline: {args.kfold}-fold CV on LABELED data (after merging) ===")
    print(f"Trying k in: {neighbors_list}  | weights='{args.weights}'\n")

    for k in neighbors_list:
        fold_acc = []
        fold_f1 = []
        fold_manh = []
        fold_w1 = []
        fold_w2 = []
        fold_w3 = []

        for _, (tr, te) in enumerate(skf.split(X_lab, y_lab), start=1):
            model = make_knn_model(n_neighbors=k, weights=args.weights)
            model.fit(X_lab[tr], y_lab[tr])

            pred = model.predict(X_lab[te])
            fold_acc.append(accuracy_score(y_lab[te], pred))
            fold_f1.append(f1_score(y_lab[te], pred, average="macro"))
            fold_manh.append(mean_manhattan(y_lab[te], pred))
            fold_w1.append(within_k_steps(y_lab[te], pred, 1))
            fold_w2.append(within_k_steps(y_lab[te], pred, 2))
            fold_w3.append(within_k_steps(y_lab[te], pred, 3))

        mean_acc = float(np.mean(fold_acc))
        mean_f1 = float(np.mean(fold_f1))
        mean_m = float(np.mean(fold_manh))

        print(f"k={k:>2} | acc={mean_acc:.4f} | macroF1={mean_f1:.4f} | manhattan={mean_m:.4f} "
              f"| within1={np.mean(fold_w1):.3f} within2={np.mean(fold_w2):.3f} within3={np.mean(fold_w3):.3f}")

        score = (mean_m, mean_f1)
        if score < best_score:
            best_score = score
            best_k = k

    print(f"\n=== Selected k ===")
    print(f"Best k={best_k} by (lowest mean Manhattan, then highest Macro-F1). "
          f"Score: manhattan={best_score[0]:.4f}, macroF1={best_score[1]:.4f}")

    # Train final deploy model on ALL labeled (after merging)
    final_model = make_knn_model(n_neighbors=best_k, weights=args.weights)
    final_model.fit(X_lab, y_lab)

    # Predict ALL scans (labeled + unlabeled)
    X_all = wide_all[feature_cols].values
    all_pred = final_model.predict(X_all)

    # Confidence: kNN predict_proba exists -> take max prob
    all_probs = final_model.predict_proba(X_all)
    all_conf = all_probs.max(axis=1)

    preds_df = pd.DataFrame({
        "scan_id": wide_all["scan_id"].values,
        "pred_loc": all_pred,
        "pred_conf": all_conf,
    })

    write_predictions(conn, preds_df, args.model_version)
    print(f"\nWrote {len(preds_df)} predictions to loc_predictions (model_version='{args.model_version}')")

    conn.close()


if __name__ == "__main__":
    main()
