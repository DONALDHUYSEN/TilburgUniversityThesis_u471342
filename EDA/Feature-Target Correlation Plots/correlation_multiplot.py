import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 


#=======================
#  All code has been improved with CLAUDE-Sonnet-4.6
#================







path = "btc_full_feature_set_daily.csv"
df = pd.read_csv(path)
 
# Ensure Date column is parsed correctly
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
 
# =========================
# Create forecast targets
# =========================
horizons = {
    "t1":  1,
    "t7":  7,
    "t30": 30,
}
 
log_close = np.log(df["Close"])
 
for label, horizon in horizons.items():
    df[f"target_{label}"] = log_close.diff().shift(-horizon)
 
# =========================
# Create output folder
# =========================
os.makedirs("outputs", exist_ok=True)
 
# =========================
# Generate one plot per horizon
# =========================
for label in horizons:
    target_col = f"target_{label}"
 
    # Select numeric columns, drop NaNs for this target
    df_numeric = df.select_dtypes(include=[np.number]).dropna(subset=[target_col])
    df_numeric = df_numeric.dropna()
 
    # Pearson correlation with the target, excluding the target itself
    target_cols_all = [f"target_{h}" for h in horizons]
    feature_cols = [c for c in df_numeric.columns if c not in target_cols_all]
 
    corr = df_numeric[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_sorted = corr.sort_values(ascending=False)
 
    colors = ["steelblue" if x > 0 else "indianred" for x in corr_sorted]
 
    fig, ax = plt.subplots(figsize=(12, 20))
 
    ax.barh(corr_sorted.index, corr_sorted.values, color=colors)
    ax.set_xlabel("Pearson Correlation with Future Return")
    ax.set_ylabel("Features")
    ax.set_title(f"Feature Correlation with BTC Future Daily Return ({label.upper()} horizon)")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
 
    plt.tight_layout()
 
    out_path = os.path.join("outputs", f"correlation_{label}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
 
    print(f"Saved: {out_path}")
 
print("All plots saved to outputs/")
