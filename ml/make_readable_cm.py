import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DB = "data/wiliot.db"
MODEL_VERSION = "logreg_v1"
TOP_N = 15  # try 10, 15, 20

def main():
    conn = sqlite3.connect(DB)

    df = pd.read_sql_query(
        """
        SELECT s.true_loc, p.pred_loc
        FROM scans s
        JOIN loc_predictions p ON p.scan_id = s.scan_id
        WHERE s.true_loc IS NOT NULL
          AND p.model_version = ?
        """,
        conn,
        params=(MODEL_VERSION,),
    )
    conn.close()

    df["true_loc"] = df["true_loc"].astype(str)
    df["pred_loc"] = df["pred_loc"].astype(str)

    # Pick the TOP_N most frequent true labels (so it’s not dominated by rare classes)
    top_labels = df["true_loc"].value_counts().head(TOP_N).index.tolist()

    # Filter to only those labels (both true and predicted),
    # and bucket everything else into "OTHER" to keep it clean.
    def bucket(x: str) -> str:
        return x if x in top_labels else "OTHER"

    y_true = df["true_loc"].map(bucket)
    y_pred = df["pred_loc"].map(bucket)

    labels = top_labels + ["OTHER"]

    # --- Plot 1: raw counts ---
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    disp1 = ConfusionMatrixDisplay(cm_counts, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp1.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix (Counts) — Top {TOP_N} + OTHER")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix_topN_counts.png", dpi=200)
    plt.close()

    # --- Plot 2: normalized by true label (row-wise %) ---
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp2.plot(xticks_rotation=45, values_format=".2f")
    plt.title(f"Confusion Matrix (Row-normalized) — Top {TOP_N} + OTHER")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix_topN_normalized.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" - figures/confusion_matrix_topN_counts.png")
    print(" - figures/confusion_matrix_topN_normalized.png")

if __name__ == "__main__":
    main()
