"""
Diebold-Mariano Test & Model Confidence Set (MCS)
Horizon: t+30
Models:  ARIMA, XGBoost, LSTM, CNN-LSTM (all post-hyperparameter tuning)

What this script does
─────────────────────
1. Loads the four prediction CSVs and aligns them on the shared Date index.
2. Computes per-observation squared errors (SE) for each model.
3. Runs the Diebold-Mariano (DM) test for every pair (6 combinations).
   - Uses the Harvey, Leybourne & Newbold (1997) small-sample correction.
   - Loss function: squared error  →  tests equal MSE accuracy.
4. Runs the Model Confidence Set (MCS) procedure (Tmax statistic, 5 000
   bootstrap replications) to identify the set of models that cannot be
   statistically distinguished from the best at α = 0.10 and α = 0.05.
5. Prints a clean summary to the terminal.

Dependencies
────────────
    pip install arch statsmodels scipy pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
from scipy import stats

# MCS implementation (arch package) 
try:
    from arch.bootstrap import MCS
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("[WARNING] 'arch' package not found – MCS section will be skipped.")
    print("          Install with:  pip install arch\n")



# 1.  FILE PATHS
FILES = {
    "ARIMA":    "arima_test_predictions_30.csv",
    "XGBoost":  "xgboost_test_predictions_t30.csv",
    "LSTM":     "lstm_test_predictions_random_search_30.csv",
    "CNN-LSTM": "cnnlstm_test_predictions_after_hyper_t30.csv",
}

# Column names that hold actual and predicted log-returns in each CSV
# (discovered from the file headers)
ACTUAL_COL = {
    "ARIMA":    "actual_log_return",
    "XGBoost":  "actual_log_return",
    "LSTM":     "actual_log_return_t_plus_30",
    "CNN-LSTM": "actual_log_return_t_plus_30",
}
PRED_COL = {
    "ARIMA":    "predicted_log_return_t_plus_30",
    "XGBoost":  "predicted_log_return",
    "LSTM":     "predicted_log_return_t_plus_30",
    "CNN-LSTM": "predicted_log_return_t_plus_30",
}







# 2.  DIEBOLD-MARIANO TEST
#     Harvey, Leybourne & Newbold (1997) small-sample corrected version.
#     Loss differential: d_t = e1_t^2 – e2_t^2  (squared-error loss)
#     H0: E[d_t] = 0  (equal predictive accuracy)
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 30) -> tuple[float, float, str]:
    """
    Diebold-Mariano test with HLN small-sample correction.

    Parameters
    ----------
    e1, e2 : forecast error arrays for model 1 and model 2
    h      : forecast horizon (30 for t+30)

    Returns
    -------
    dm_stat : HLN-corrected DM statistic
    p_value : two-sided p-value
    verdict : plain-English result string
    """
    d = e1 ** 2 - e2 ** 2          # loss differential (squared-error)
    T = len(d)
    d_bar = np.mean(d)

    # Autocovariance sum up to lag h-1
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T

    if var_d <= 0:
        return np.nan, np.nan, "Variance of loss differential ≤ 0 – cannot compute"

    dm_raw = d_bar / np.sqrt(var_d)

    # HLN correction factor
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm_raw * hln

    # t-distribution with T-1 degrees of freedom
    p_value = 2 * stats.t.sf(abs(dm_stat), df=T - 1)

    if p_value < 0.01:
        sig = "*** (p<0.01)"
    elif p_value < 0.05:
        sig = "**  (p<0.05)"
    elif p_value < 0.10:
        sig = "*   (p<0.10)"
    else:
        sig = "    (not significant)"

    better = "Model 1" if d_bar > 0 else "Model 2"   # lower loss = better
    verdict = f"{sig}  →  {better} is more accurate"
    return dm_stat, p_value, verdict







# 3.  LOAD DATA
def load_errors(files: dict, actual_col: dict, pred_col: dict) -> pd.DataFrame:
    """Load each CSV, extract errors, align on Date."""
    series = {}
    for name, path in files.items():
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.dropna(subset=[actual_col[name], pred_col[name]])
        err = df.set_index("Date")[actual_col[name]] - df.set_index("Date")[pred_col[name]]
        series[name] = err.rename(name)

    errors = pd.concat(series.values(), axis=1).dropna()
    return errors






# 4.  MAIN
def main():
    DIVIDER  = "=" * 72
    DIVIDER2 = "-" * 72

    print(f"\n{DIVIDER}")
    print("  SIGNIFICANCE TESTING  –  Bitcoin Log-Return Forecasting (t+30)")
    print(f"{DIVIDER}\n")

    # Load
    print("Loading prediction files …")
    errors = load_errors(FILES, ACTUAL_COL, PRED_COL)
    models = list(errors.columns)
    N = len(errors)
    print(f"  Aligned test observations : {N}")
    print(f"  Date range               : {errors.index.min().date()} → {errors.index.max().date()}")
    print(f"  Models                   : {', '.join(models)}\n")

    # Descriptive error stats 
    print(DIVIDER2)
    print("  FORECAST ERROR SUMMARY (test set)")
    print(DIVIDER2)
    summary = pd.DataFrame({
        "MAE":  errors.abs().mean(),
        "RMSE": np.sqrt((errors ** 2).mean()),
        "Bias": errors.mean(),
    })
    print(summary.to_string(float_format="{:.6f}".format))
    print()

    # Diebold-Mariano
    print(DIVIDER)
    print("  DIEBOLD-MARIANO TEST  (HLN small-sample correction, h=30)")
    print("  Loss function: Squared Error  |  H0: equal predictive accuracy")
    print(DIVIDER)
    print(f"  {'Pair':<28} {'DM Stat':>9} {'p-value':>9}  Result")
    print(DIVIDER2)

    pairs = list(itertools.combinations(models, 2))
    for m1, m2 in pairs:
        e1 = errors[m1].values
        e2 = errors[m2].values
        dm_stat, p_val, verdict = dm_test(e1, e2, h=30)
        pair_label = f"{m1}  vs  {m2}"
        if np.isnan(dm_stat):
            print(f"  {pair_label:<28}  {verdict}")
        else:
            print(f"  {pair_label:<28} {dm_stat:>9.4f} {p_val:>9.4f}  {verdict}")

    print(DIVIDER2)
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("  Positive DM stat → Model 1 has higher loss (Model 2 is better)")
    print()

    # Model Confidence Set 
    if not HAS_ARCH:
        print("[SKIPPED] MCS requires the 'arch' package.\n")
        return

    print(DIVIDER)
    print("  MODEL CONFIDENCE SET (MCS)")
    print("  Tmax statistic  |  5 000 bootstrap replications")
    print("  Loss function: Squared Error")
    print(DIVIDER)

    losses = errors ** 2          # (T × 4) squared-error loss matrix

    for alpha, label in [(0.10, "α = 0.10"), (0.05, "α = 0.05")]:
        mcs = MCS(losses, size=alpha, reps=5000, seed=42, method="max")
        mcs.compute()

        included = mcs.included
        excluded = mcs.excluded

        print(f"\n  [{label}]")
        print(f"  Superior Model Set  : {', '.join(included) if len(included) else '(none)'}")
        print(f"  Eliminated Models   : {', '.join(excluded) if len(excluded) else '(none)'}")

        print(f"\n  MCS p-values ({label}):")
        pvals_df = mcs.pvalues
        # pvalues may be a DataFrame or Series depending on arch version
        if isinstance(pvals_df, pd.DataFrame):
            pvals = pvals_df.iloc[:, 0].sort_values(ascending=False)
        else:
            pvals = pvals_df.sort_values(ascending=False)
        for model_name, pv in pvals.items():
            tag = "  ← IN superior set" if model_name in included else ""
            print(f"    {model_name:<12} p = {pv:.4f}{tag}")

    print(f"\n{DIVIDER}")
    print("  INTERPRETATION GUIDE")
    print(DIVIDER)
    print("""
  DM Test
  ───────
  • A significant result (p < 0.05) means the two models have
    statistically different forecast accuracy.
  • The sign of the DM statistic tells you which is better:
    positive → Model 2 wins; negative → Model 1 wins.

  Model Confidence Set
  ────────────────────
  • The MCS is the smallest set of models that cannot be
    statistically distinguished from the best model at level α.
  • Models eliminated from the MCS are significantly worse than
    at least one model still in the set.
  • Cite the MCS result as your definitive ranking claim:
    e.g., "CNN-LSTM is in the MCS at α=0.05, confirming it
    cannot be ruled out as the best-performing model."
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
