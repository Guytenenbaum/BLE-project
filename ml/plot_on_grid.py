import argparse
import sqlite3
import re
from typing import Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Location parsing: "C7" -> (x, y)
# -----------------------------
def loc_to_xy(loc: str) -> Optional[Tuple[int, int]]:
    if loc is None:
        return None
    s = str(loc).strip().upper()
    m = re.match(r"^([A-Z]+)\s*(\d+)$", s)
    if not m:
        return None

    letters = m.group(1)
    y = int(m.group(2))

    # Excel-like letters: A=1, B=2 ... Z=26, AA=27...
    x = 0
    for ch in letters:
        x = x * 26 + (ord(ch) - ord("A") + 1)

    return x, y


def xy_to_loc(x: int) -> str:
    # 1 -> A, 2 -> B, ... 26 -> Z, 27 -> AA
    out = []
    while x > 0:
        x, r = divmod(x - 1, 26)
        out.append(chr(r + ord("A")))
    return "".join(reversed(out))


def manhattan(true_loc: str, pred_loc: str) -> Optional[int]:
    a = loc_to_xy(true_loc)
    b = loc_to_xy(pred_loc)
    if a is None or b is None:
        return None
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -----------------------------
# DB fetch
# -----------------------------
def load_sample(conn: sqlite3.Connection, model_version: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT s.scan_id, s.ts, s.true_loc, p.pred_loc, p.pred_conf
        FROM scans s
        JOIN loc_predictions p ON p.scan_id = s.scan_id
        WHERE s.true_loc IS NOT NULL
          AND p.model_version = ?
        ORDER BY s.scan_id;
        """,
        conn,
        params=(model_version,),
    )

    if df.empty:
        raise RuntimeError("No labeled rows with predictions found. Check --model-version and DB path.")

    # Parse coords
    df["true_xy"] = df["true_loc"].apply(loc_to_xy)
    df["pred_xy"] = df["pred_loc"].apply(loc_to_xy)
    df = df[df["true_xy"].notna() & df["pred_xy"].notna()].copy()

    if df.empty:
        raise RuntimeError("Could not parse any true/pred locations into grid coordinates (like 'C7').")

    df["true_x"] = df["true_xy"].apply(lambda t: t[0])
    df["true_y"] = df["true_xy"].apply(lambda t: t[1])
    df["pred_x"] = df["pred_xy"].apply(lambda t: t[0])
    df["pred_y"] = df["pred_xy"].apply(lambda t: t[1])
    df["manhattan"] = [
        manhattan(t, p) for t, p in zip(df["true_loc"].astype(str), df["pred_loc"].astype(str))
    ]

    # Sample n rows reproducibly
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].reset_index(drop=True)


# -----------------------------
# Plot
# -----------------------------
def plot_grid(df: pd.DataFrame, out_path: str, title: str = ""):
    # Determine grid bounds from the sample + small padding
    xs = np.concatenate([df["true_x"].to_numpy(), df["pred_x"].to_numpy()])
    ys = np.concatenate([df["true_y"].to_numpy(), df["pred_y"].to_numpy()])

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    pad = 1
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    # Small jitter so overlapping points are visible
    jitter = 0.08
    jtx = (np.random.rand(len(df)) - 0.5) * 2 * jitter
    jty = (np.random.rand(len(df)) - 0.5) * 2 * jitter

    true_x = df["true_x"].to_numpy() + jtx
    true_y = df["true_y"].to_numpy() + jty
    pred_x = df["pred_x"].to_numpy() - jtx
    pred_y = df["pred_y"].to_numpy() - jty

    mean_m = float(np.nanmean(df["manhattan"].to_numpy()))

    plt.figure(figsize=(10, 6))

    # Grid lines
    for x in range(min_x, max_x + 1):
        plt.axvline(x, linewidth=0.5, alpha=0.2)
    for y in range(min_y, max_y + 1):
        plt.axhline(y, linewidth=0.5, alpha=0.2)

    # Connect true -> pred
    for i in range(len(df)):
        plt.plot([true_x[i], pred_x[i]], [true_y[i], pred_y[i]], linewidth=1, alpha=0.7)

    # True and Pred markers
    plt.scatter(true_x, true_y, marker="o", s=70, label="True")
    plt.scatter(pred_x, pred_y, marker="x", s=70, label="Predicted")

    # Annotate points with index (1..n) and manhattan distance
    for i in range(len(df)):
        d = df.loc[i, "manhattan"]
        plt.text(true_x[i] + 0.12, true_y[i] + 0.12, f"{i+1}", fontsize=9)
        plt.text(pred_x[i] + 0.12, pred_y[i] - 0.18, f"d={d}", fontsize=8, alpha=0.85)

    # Axes ticks: show letters on x
    x_ticks = list(range(min_x, max_x + 1))
    plt.xticks(x_ticks, [xy_to_loc(x) for x in x_ticks])
    plt.yticks(list(range(min_y, max_y + 1)))

    plt.xlim(min_x - 0.5, max_x + 0.5)
    plt.ylim(min_y - 0.5, max_y + 0.5)

    ttl = title.strip()
    if not ttl:
        ttl = f"True vs Predicted locations (n={len(df)}) | mean Manhattan={mean_m:.2f}"
    plt.title(ttl)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to SQLite DB (e.g., data/wiliot.db)")
    parser.add_argument("--model-version", required=True, help="model_version from loc_predictions")
    parser.add_argument("--n", type=int, default=20, help="Number of labeled points to sample and plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out", default="figures/true_vs_pred_20.png", help="Output PNG path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    df = load_sample(conn, args.model_version, args.n, args.seed)
    conn.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_grid(df, args.out)
    print(f"Saved: {args.out}")
    print(df[["scan_id", "ts", "true_loc", "pred_loc", "pred_conf", "manhattan"]].to_string(index=False))


if __name__ == "__main__":
    main()
