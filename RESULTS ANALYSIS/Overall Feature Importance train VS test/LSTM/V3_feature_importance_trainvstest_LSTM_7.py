import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import tensorflow as tf

# === Settings ===
train_csv_path = "train_t7.csv"
test_csv_path = "test_t7.csv"
model_path = "lstm_model_t7.keras"
scaler_path = "scaler_tv_t7.joblib"
feature_cols_path = "feature_cols_t7.txt"
output_path = "V3_LSTM_SHAP_train_vs_test_t7_normalized.png"

LOOKBACK = 100
BACKGROUND_SIZE = 100
MAX_SHAP_SAMPLES_TRAIN = 500
MAX_SHAP_SAMPLES_TEST = 500

title = "Normalized SHAP Feature Importance: LSTM Train vs Test Horizon 7"

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

# === Helper: build LSTM sequences ===
def build_sequences(X, lookback):
    sequences = []
    for i in range(lookback, len(X)):
        sequences.append(X[i - lookback:i])
    return np.array(sequences, dtype=np.float32)

# === Helper: calculate SHAP importance ===
def calculate_lstm_shap_importance(model, X_seq, feature_cols, background_size, max_samples):
    if len(X_seq) > max_samples:
        X_seq_shap = X_seq[-max_samples:]
    else:
        X_seq_shap = X_seq

    background = X_seq_shap[:background_size]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_seq_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if shap_values.ndim == 4:
        shap_values = shap_values[:, :, :, 0]

    importance = np.abs(shap_values).mean(axis=(0, 1))

    return pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    })

# === Load artefacts ===
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

with open(feature_cols_path, "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

print(f"Number of model features: {len(feature_cols)}")

# === Scale features using saved scaler ===
X_train_raw = train_df[feature_cols].values.astype(np.float32)
X_test_raw = test_df[feature_cols].values.astype(np.float32)

X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

# === Build LSTM sequences ===
X_train_seq = build_sequences(X_train_scaled, LOOKBACK)
X_test_seq = build_sequences(X_test_scaled, LOOKBACK)

print(f"Train sequence shape: {X_train_seq.shape}")
print(f"Test sequence shape: {X_test_seq.shape}")

# === Calculate SHAP importance ===
train_imp = calculate_lstm_shap_importance(
    model=model,
    X_seq=X_train_seq,
    feature_cols=feature_cols,
    background_size=BACKGROUND_SIZE,
    max_samples=MAX_SHAP_SAMPLES_TRAIN
)

test_imp = calculate_lstm_shap_importance(
    model=model,
    X_seq=X_test_seq,
    feature_cols=feature_cols,
    background_size=BACKGROUND_SIZE,
    max_samples=MAX_SHAP_SAMPLES_TEST
)

# === Merge train and test importance ===
importance_df = train_imp.merge(
    test_imp,
    on="feature",
    suffixes=("_train", "_test")
)

# === Filter to selected non-lagged features ===
importance_df = importance_df[
    importance_df["feature"].isin(wanted_features)
].copy()

# === Normalise using combined max ===
max_value = max(
    importance_df["importance_train"].max(),
    importance_df["importance_test"].max()
)

importance_df["train_norm"] = importance_df["importance_train"] / max_value
importance_df["test_norm"] = importance_df["importance_test"] / max_value

# Train left, test right
importance_df["train_plot"] = -importance_df["train_norm"]
importance_df["test_plot"] = importance_df["test_norm"]

importance_df["combined"] = (
    importance_df["train_norm"] + importance_df["test_norm"]
)

importance_df = importance_df.sort_values("combined", ascending=True)

print(f"Selected features shown: {len(importance_df)}")

# === Plot ===
plt.figure(figsize=(11, len(importance_df) * 0.12))

plt.barh(
    importance_df["feature"],
    importance_df["train_plot"],
    height=0.4,
    label="Train"
)

plt.barh(
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