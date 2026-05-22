import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf




#=======================
#  All code has been improved with CLAUDE-Sonnet-4.6
#================








# =========================
# Configuration
# =========================
DATA_PATH   = "btc_full_feature_set_daily.csv"
OUTPUT_DIR  = "outputs"
TRAIN_RATIO = 0.80          # 80 % train / 20 % test (chronological split)
ACF_LAGS    = 40            # number of lags shown in ACF plot

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load & prepare data
# =========================
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Bitcoin log returns (not shifted – used as the base series)
log_ret = np.log(df["Close"]).diff()

# =========================
# Define forecast targets
# horizon h  →  log return that materialises h days ahead
# =========================
horizons = {"t1": 1, "t7": 7, "t30": 30}

for label, h in horizons.items():
    df[f"target_{label}"] = log_ret.shift(-h)

# =========================
# Feature columns
# – exclude Date, Close-derived targets, and non-numeric columns
# =========================
target_cols = [f"target_{l}" for l in horizons]
exclude_cols = {"Date"} | set(target_cols)

feature_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols
]

# =========================
# Helper – fit OLS, return residuals + DW statistic
# =========================
def analyse_horizon(label: str) -> None:
    target_col = f"target_{label}"

    # Drop rows where the target or any feature is NaN
    cols_needed = feature_cols + [target_col]
    df_clean = df[cols_needed].dropna()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # Chronological train / test split (no shuffling)
    split_idx = int(len(df_clean) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardise features (fit scaler on train only to avoid data leakage)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Fit OLS (LinearRegression uses the closed-form OLS solution)
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Predict on test set and compute residuals
    y_pred     = model.predict(X_test_sc)
    residuals  = y_test - y_pred       # actual − predicted

    # ---- Durbin-Watson statistic ----------------------------------------
    # DW ≈ 2  →  little or no first-order autocorrelation
    # DW < 2  →  positive autocorrelation in residuals
    # DW > 2  →  negative autocorrelation in residuals
    dw = durbin_watson(residuals)

    print(f"\n{'='*50}")
    print(f" Horizon : {label.upper()}")
    print(f" Test observations : {len(residuals)}")
    print(f" Durbin-Watson statistic : {dw:.4f}")
    if dw < 1.5:
        interpretation = "Positive autocorrelation detected – model misses temporal structure."
    elif dw > 2.5:
        interpretation = "Negative autocorrelation detected – possible over-differencing."
    else:
        interpretation = "No strong first-order autocorrelation (DW near 2)."
    print(f" Interpretation : {interpretation}")

    # ---- ACF plot of residuals ------------------------------------------
    # ACF captures higher-order autocorrelation that DW does not test.
    fig, ax = plt.subplots(figsize=(12, 4))

    plot_acf(residuals, lags=ACF_LAGS, alpha=0.05, ax=ax, color="steelblue",
             vlines_kwargs={"colors": "steelblue"})

    ax.set_title(
        f"ACF of OLS Residuals – {label.upper()} Horizon\n"
        f"(Durbin-Watson = {dw:.4f}  |  {interpretation})",
        fontsize=11
    )
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"acf_residuals_{label}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" ACF plot saved → {out_path}")


# =========================
# Run analysis for each horizon
# =========================
for label in horizons:
    analyse_horizon(label)

print("\nDone. All outputs saved to the 'outputs' folder.")
