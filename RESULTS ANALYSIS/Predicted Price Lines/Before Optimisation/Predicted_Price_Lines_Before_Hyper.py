import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Instellingen
# =========================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "ARIMA t+1": "arima_test_predictions.csv",
    "ARIMA t+7": "arima_test_predictions_7.csv",
    "ARIMA t+30": "arima_test_predictions_30.csv",

    "XGBoost t+1": "xgboost_test_predictions_1.csv",
    "XGBoost t+7": "xgboost_test_predictions_7.csv",
    "XGBoost t+30": "xgboost_test_predictions_30.csv",

    "LSTM t+1": "lstm_test_predictions_before_hyper_1.csv",
    "LSTM t+7": "lstm_test_predictions_before_hyper_7.csv",
    "LSTM t+30": "lstm_test_predictions_before_hyper_30.csv",

    "CNN-LSTM t+1": "cnnlstm_test_predictions_before_hyper_1.csv",
    "CNN-LSTM t+7": "cnnlstm_test_predictions_before_hyper_7.csv",
    "CNN-LSTM t+30": "cnnlstm_test_predictions_before_hyper_30.csv",
}

DATE_COL = "Date"
PRICE_COL = "Close"

# =========================
# Functie om juiste prediction kolom te vinden
# =========================

def get_prediction_column(file_path):
    """
    Detecteert automatisch:
    _1.csv  -> predicted_log_return_t_plus_1
    _7.csv  -> predicted_log_return_t_plus_7
    _30.csv -> predicted_log_return_t_plus_30
    """

    match = re.search(r'_(\d+)\.csv$', file_path)

    if match:
        horizon = match.group(1)
    else:
        # fallback voor ARIMA of bestanden zonder suffix
        horizon = "1"

    return f"predicted_log_return_t_plus_{horizon}"

# =========================
# Functie: reconstruct predicted prices
# =========================

def reconstruct_predicted_price(df, pred_col):
    """
    Rekent log returns terug naar prijsniveau.

    predicted_price_t =
        start_price * exp(cumulative_sum(log_returns))
    """

    df = df.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

    df = df.dropna(subset=[DATE_COL, PRICE_COL, pred_col]).reset_index(drop=True)

    start_price = df[PRICE_COL].iloc[0]

    # cumulatieve reconstructie
    cumulative_returns = df[pred_col].cumsum()

    df["predicted_price_usd"] = start_price * np.exp(cumulative_returns)

    return df

# =========================
# Grafieken genereren
# =========================

for model_name, file_path in FILES.items():

    if not os.path.exists(file_path):
        print(f"Bestand niet gevonden: {file_path}")
        continue

    pred_col = get_prediction_column(file_path)

    df = pd.read_csv(file_path)

    required_cols = [DATE_COL, PRICE_COL, pred_col]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"{model_name} overgeslagen.")
        print(f"Ontbrekende kolommen: {missing_cols}")
        continue

    df_plot = reconstruct_predicted_price(df, pred_col)

    # =========================
    # Plot
    # =========================

    plt.figure(figsize=(14, 6))

    plt.plot(
        df_plot[DATE_COL],
        df_plot[PRICE_COL],
        label="Actual Price",
        linewidth=2,
        color="black"
    )

    plt.plot(
        df_plot[DATE_COL],
        df_plot["predicted_price_usd"],
        label="Predicted Price",
        linestyle="--",
        linewidth=2,
        color="green"
    )

    plt.title(f"Reconstructed Bitcoin Price vs Predicted ({model_name})")

    plt.xlabel("Time")
    plt.ylabel("Bitcoin Price (USD)")

    plt.legend()
    plt.grid(True, alpha=0.4)

    plt.tight_layout()

    # veilige bestandsnaam
    safe_model_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_model_name}_predicted_price.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"Opgeslagen: {output_path}")

print("\nKlaar. Alle grafieken staan in de map 'outputs'.")
