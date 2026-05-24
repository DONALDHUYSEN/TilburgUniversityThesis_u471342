import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# === Settings ===
train_csv_path = "train_t7.csv"
test_csv_path = "test_t7.csv"
model_path = "xgb_model_t7.json"
output_path = "XGBoost_SHAP_train_vs_test_t7_normalized.png"

title = "Normalized SHAP Feature Importance: XGBoost Train vs Test Horizon 7"

wanted_features = [
    "Volume","SMA_7","SMA_21","SMA_50","SMA_200","RET_30D","Market_Regime",
    "SMA_7_adjusted","SMA_21_adjusted","SMA_50_adjusted","SMA_200_adjusted",
    "MACD","MACD_signal","MACD_hist","MACD_adjusted","MACD_signal_adjusted",
    "MACD_hist_adjusted","UpperBB","MiddleBB","LowerBB","UpperBB_adjusted",
    "MiddleBB_adjusted","LowerBB_adjusted","ATR14","Stoch_K","Stoch_D","CPI",
    "GDP","Unemployment Rate","10-Year Treasury Rate","Industrial Production",
    "M2 Money Stock","Consumer Confidence","Corporate Bond Spread",
    "Crude Oil Prices","Effective Federal Funds Rate","Housing Starts",
    "Personal Consumption Expenditures","Total Nonfarm Payrolls",
    "Real Personal Income","Total Vehicle Sales","Retail Sales",
    "30-Year Fixed Mortgage Rate","15-Year Fixed Mortgage Rate","CPI_adjusted",
    "GDP_adjusted","Unemployment Rate_adjusted","10-Year Treasury Rate_adjusted",
    "Industrial Production_adjusted","M2 Money Stock_adjusted",
    "Consumer Confidence_adjusted","Corporate Bond Spread_adjusted",
    "Crude Oil Prices_adjusted","Effective Federal Funds Rate_adjusted",
    "Housing Starts_adjusted","Personal Consumption Expenditures_adjusted",
    "Total Nonfarm Payrolls_adjusted","Real Personal Income_adjusted",
    "Total Vehicle Sales_adjusted","Retail Sales_adjusted",
    "30-Year Fixed Mortgage Rate_adjusted","15-Year Fixed Mortgage Rate_adjusted",
    "sp500_open","sp500_high","sp500_low","sp500_close","sp500_volume",
    "vix_open","vix_high","vix_low","vix_close","dxy_open","dxy_high",
    "dxy_low","dxy_close","nikkei_open","nikkei_high","nikkei_low",
    "nikkei_close","nikkei_volume","sp500_volume_adjusted",
    "vix_open_adjusted","vix_close_adjusted","dxy_open_adjusted",
    "dxy_close_adjusted","nikkei_close_adjusted","AdrActCnt",
    "TxCnt","AdrBalCnt","HashRate","SplyCur"
]

# === Load data ===
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# === Reconstruct model features ===
exclude_cols = {"Date", "Close", "log_return", "asset"}

feature_cols = [
    col for col in train_df.columns
    if col not in exclude_cols and not col.startswith("target_")
]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"Number of model features: {len(feature_cols)}")

# === Load model ===
model = xgb.XGBRegressor()
model.load_model(model_path)

# === SHAP values ===
explainer = shap.TreeExplainer(model)

shap_train = explainer.shap_values(X_train)
shap_test = explainer.shap_values(X_test)

train_importance = np.abs(shap_train).mean(axis=0)
test_importance = np.abs(shap_test).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "train_importance": train_importance,
    "test_importance": test_importance
})

# === Filter to selected non-lagged features ===
importance_df = importance_df[
    importance_df["feature"].isin(wanted_features)
].copy()

# === Normalise using combined max ===
max_value = max(
    importance_df["train_importance"].max(),
    importance_df["test_importance"].max()
)

importance_df["train_norm"] = importance_df["train_importance"] / max_value
importance_df["test_norm"] = importance_df["test_importance"] / max_value

# Train to the left, test to the right
importance_df["train_plot"] = -importance_df["train_norm"]
importance_df["test_plot"] = importance_df["test_norm"]

# Sort by combined importance
importance_df["combined"] = (
    importance_df["train_norm"] + importance_df["test_norm"]
)

importance_df = importance_df.sort_values("combined", ascending=True)

print(f"Selected features shown: {len(importance_df)}")

# === Plot ===
plt.figure(figsize=(10, len(importance_df) * 0.10))

bars_train = plt.barh(
    importance_df["feature"],
    importance_df["train_plot"],
    height=0.4,
    label="Train"
)

bars_test = plt.barh(
    importance_df["feature"],
    importance_df["test_plot"],
    height=0.4,
    label="Test"
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Normalised Mean |SHAP value|", fontsize=10)
plt.ylabel("Features", fontsize=9, labelpad=0)
plt.title(title, fontsize=12)

plt.yticks(fontsize=7)
plt.xticks(fontsize=8)
plt.legend(fontsize=8)

plt.grid(axis="x", alpha=0.3)

# Add value labels
for _, row in importance_df.iterrows():
    plt.text(
        row["train_plot"],
        row["feature"],
        f"{row['train_norm']:.2f}",
        va="center",
        ha="right",
        fontsize=5
    )
    plt.text(
        row["test_plot"],
        row["feature"],
        f"{row['test_norm']:.2f}",
        va="center",
        ha="left",
        fontsize=5
    )

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {output_path}")
