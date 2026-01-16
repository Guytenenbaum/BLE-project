import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DB = "data/wiliot.db"
MODEL_VERSION = "logreg_v1"

conn = sqlite3.connect(DB)

df = pd.read_sql_query(
    """
    SELECT s.true_loc, p.pred_loc
    FROM scans s
    JOIN loc_predictions p ON p.scan_id = s.scan_id
    WHERE s.true_loc IS NOT NULL AND p.model_version = ?
    """,
    conn, params=(MODEL_VERSION,)
)

y_true = df["true_loc"].astype(str).values
y_pred = df["pred_loc"].astype(str).values

cm = confusion_matrix(y_true, y_pred, labels=sorted(df["true_loc"].unique()))
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(df["true_loc"].unique()))
disp.plot(xticks_rotation=90)
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=200)
print("Saved figures/confusion_matrix.png")
