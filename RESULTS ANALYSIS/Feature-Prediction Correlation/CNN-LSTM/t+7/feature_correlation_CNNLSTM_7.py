import pandas as pd
import matplotlib.pyplot as plt

# === Settings ===
csv_path = "cnnlstm_test_predictions_after_hyper_t7.csv"
prediction_col = "predicted_log_return_t_plus_7"
output_path = "feature_correlation_barplot_CNNLSTM_FvP_t7.png"


include_lagged_features = False

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

df = pd.read_csv(csv_path)

allowed_features = wanted_features.copy()

if include_lagged_features:
    allowed_features += [f"{feature}_lag1" for feature in wanted_features]

available_features = [
    col for col in allowed_features
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
]

missing_features = [col for col in allowed_features if col not in df.columns]

print(f"Number of selected features found: {len(available_features)}")
print(f"Number of selected features missing: {len(missing_features)}")

correlations = df[available_features].corrwith(df[prediction_col]).dropna()
correlations = correlations.sort_values(ascending=True)

plt.figure(figsize=(12, max(8, len(correlations) * 0.28)))

bars = plt.barh(correlations.index, correlations.values)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.title("CNN-LSTM Feature Correlations with Prediction Horizon 7")

for bar in bars:
    value = bar.get_width()
    plt.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()