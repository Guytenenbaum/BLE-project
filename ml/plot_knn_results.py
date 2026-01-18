import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt
# Results you printed
ks = [3, 5, 7, 9, 11, 15]
acc = [0.3155, 0.3028, 0.3028, 0.3007, 0.2979, 0.3007]
macro_f1 = [0.2973, 0.2758, 0.2755, 0.2772, 0.2739, 0.2717]
manhattan = [2.5077, 2.5972, 2.6127, 2.5655, 2.5507, 2.5345]
within1 = [0.456, 0.455, 0.447, 0.444, 0.444, 0.451]
within2 = [0.592, 0.601, 0.598, 0.603, 0.606, 0.611]
within3 = [0.700, 0.699, 0.702, 0.706, 0.705, 0.708]
best_k = 3

# Project root = parent of the folder that contains this script (ml/)
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

best_idx = ks.index(best_k)
vals = [within1[best_idx], within2[best_idx], within3[best_idx]]

plt.figure()
plt.bar(["Within-1", "Within-2", "Within-3"], vals)
plt.ylim(0, 1)
plt.ylabel("Fraction of predictions")
plt.title(f"Localization tolerance (kNN, k={best_k})")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

out_path = FIG_DIR / "fig_within_thresholds.png"
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved: {out_path}")