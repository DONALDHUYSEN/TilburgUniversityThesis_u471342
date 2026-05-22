"""
V3_lstm_btc_log_returns_randomsearch_1_after_hyper.py
──────────────────────────────────────────────────────
LSTM model for Bitcoin log-return forecasting.
Online learning variant with a fixed 100-day lookback window.
WITH hyperparameter optimisation via RANDOM SEARCH over PARAM_GRID.

WHAT IS NEW IN V3 (vs. V2)
───────────────────────────
1. VALIDATION DATA LEAKAGE FIX (critical)
   During random search, validation is now performed by predict_only_evaluate
   instead of online_evaluate.  The new function generates predictions
   sequentially using the rolling context — identical to online_evaluate —
   but deliberately omits the train_on_batch step.  Validation targets
   therefore never influence model weights during hyperparameter search,
   eliminating the data-leakage that was present in V2.

   online_evaluate (with train_on_batch) is still used on the test set in
   fit_and_evaluate, where true online learning is the intended behaviour.

2. N_RANDOM_DRAWS REDUCED FROM 60 TO 30
   The leakage-free validation loop is slightly cheaper (no gradient update
   per step), so 30 draws still provides a reasonable search coverage while
   keeping total wall-clock time comparable to V2.

ALL V2 FUNCTIONALITY IS OTHERWISE PRESERVED UNCHANGED
──────────────────────────────────────────────────────

WHAT IS NEW IN V2 (vs. lstm_btc_log_returns_random_search_1.py)
────────────────────────────────────────────────────────────────
1. DATASET SAVING AFTER FEATURE ENGINEERING & SPLITTING
   After feature engineering and the chronological 70/15/15 split, the three
   unscaled DataFrames are saved as CSV files:
     train_t{H}.csv  /  valid_t{H}.csv  /  test_t{H}.csv
   where {H} is the forecast horizon (e.g. 1, 7, 30).

   Key guarantees:
   • Saved BEFORE any scaling — raw feature values are preserved.
   • No transformations are applied after the split, so there is zero leakage.
   • The saved columns are exactly the inputs the model uses at runtime.
   • The test split is suitable for out-of-sample SHAP / permutation importance.

2. MULTI-HORIZON COMPATIBILITY
   FORECAST_HORIZON is now a module-level constant.  The target column is
   always built as  df["log_return"].shift(-FORECAST_HORIZON),  so changing
   FORECAST_HORIZON (e.g. to 7 or 30) automatically propagates through the
   entire pipeline (feature engineering → split → file names → metrics).
   Running the script for each desired horizon produces horizon-tagged outputs.

3. SCALER SAVED FOR LATER REUSE (optional but recommended)
   After the final model is trained on train + validation data, the fitted
   StandardScaler is serialised to:
     scaler_tv_t{H}.joblib
   This allows SHAP or permutation importance to be applied later without
   re-fitting anything:
     scaler = joblib.load("scaler_tv_t1.joblib")
     X_test_scaled = scaler.transform(test_df[feature_cols].values)

4. TRAINED MODEL SAVED FOR LATER REUSE (optional but recommended)
   The final LSTM model (trained on train + validation) is saved to:
     lstm_model_t{H}.keras
   Load it later with:
     model = tf.keras.models.load_model("lstm_model_t1.keras")

5. FEATURE COLUMN LIST SAVED
   The ordered list of feature columns is written to:
     feature_cols_t{H}.txt
   (one column name per line).  This makes it trivial to reconstruct the
   exact feature matrix from the saved CSVs in a separate analysis notebook.

ALL EXISTING FUNCTIONALITY IS PRESERVED UNCHANGED
──────────────────────────────────────────────────
  • Random search hyperparameter optimisation (N_RANDOM_DRAWS combinations)
  • Online learning via train_on_batch
  • 100-day fixed lookback window
  • Walk-forward style evaluation
  • MAE / RMSE / MAPE / Directional Accuracy metrics
  • lstm_random_search_results_1.csv  (search results)
  • lstm_test_predictions_random_search_1.csv  (test predictions)

REPRODUCIBILITY WORKFLOW
────────────────────────
  1. Set FORECAST_HORIZON to the desired horizon (e.g. 1, 7, or 30).
  2. Run this script.  Outputs produced for horizon H:
       train_tH.csv            — unscaled training features + target
       valid_tH.csv            — unscaled validation features + target
       test_tH.csv             — unscaled test features + target (for SHAP)
       feature_cols_tH.txt     — ordered list of feature column names
       scaler_tv_tH.joblib     — scaler fitted on train+validation features
       lstm_model_tH.keras     — trained LSTM weights (best hyperparameters)
       lstm_random_search_results_1.csv
       lstm_test_predictions_random_search_1.csv
  3. In a separate analysis script, load the saved artefacts and run SHAP or
     permutation importance directly on test_tH.csv without retraining.

MODEL DESCRIPTION
─────────────────
A Long Short-Term Memory (LSTM) network is used for sequence-to-one
regression, forecasting the next-period log return at horizon t+H.

Online learning approach: the model is initially trained on the full
training set and subsequently updated after each new observation arrives
using Keras's train_on_batch.  This simulates a streaming data environment
in which the model adapts continuously to new information.

Architecture:
  - One or two stacked LSTM layers.
  - A dropout layer for regularisation.
  - A single dense linear output unit.
  - Adam optimiser with mean squared error (MSE) loss.

METHODOLOGY
────────────
Phase 1 — Initial training (train split):
  The scaler is fit on the training data.  The model is trained on all
  sequences derived from the scaled training set, with early stopping
  monitored on a trailing validation window carved from the training
  sequences.

Phase 2 — Random search (validation split):
  N_RANDOM_DRAWS combinations are randomly sampled from the full PARAM_GRID
  pool.  Each candidate is trained on the training split, then evaluated on
  the validation split using predict_only_evaluate: predictions are generated
  sequentially using the rolling context, but NO weight updates occur on
  validation data (train_on_batch is intentionally omitted).
  The candidate with the lowest validation RMSE is selected as best.
  The training scaler is reused for the validation split (no re-fitting).

Phase 3 — Online evaluation (test split):
  The best model is retrained from scratch on train + validation data.
  The scaler is re-fit on train + validation features only.
  Then, for each observation in the test split:
    1. Predict t+H log return using the last LOOKBACK scaled rows.
    2. Record the prediction and the realised return.
    3. Update the model weights via train_on_batch on the new observation.
"""

import logging
import os
import random
import warnings
from math import sqrt
from pathlib import Path

# ── Silence TensorFlow noise BEFORE importing TensorFlow ──────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, LSTM
from keras.models import Sequential
from keras.optimizers import Adam


# ── Reproducibility ────────────────────────────────────────────────────────────
# Fix all random seeds so results are reproducible across runs.
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH     = "btc_clean.csv"   # Path to the input data file
PRICE_COLUMN = "Close"           # Name of the price column in the CSV
DATE_COLUMN  = "Date"            # Name of the date column in the CSV

# Train / validation / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

# ── NEW (V2): Forecast horizon ─────────────────────────────────────────────────
# Change this value to run the full pipeline for a different horizon.
# Examples: 1 (t+1), 7 (t+7), 30 (t+30).
# All output filenames and the target column will automatically reflect the
# chosen horizon, so each run produces self-contained, horizon-tagged artefacts.
FORECAST_HORIZON  = 1    # Predict the log return H days ahead (t+H)

LOOKBACK          = 100  # Use the last 100 days as input to each prediction
EARLY_STOP_ROUNDS = 10   # EarlyStopping patience in epochs
EARLY_STOP_WINDOW = 30   # Number of trailing sequences held out for early stopping
MAX_EPOCHS        = 100  # Maximum number of training epochs

LAG_FEATURES   = list(range(1, 6))  # Lagged log returns: t-1 to t-5
ROLLING_WINDOW = 5                  # Window size for rolling volatility feature


# ── Hyperparameter search space ────────────────────────────────────────────────
# All possible values for each hyperparameter.
# The full grid contains 4 × 2 × 4 × 3 × 3 = 288 combinations in total.
#
# Note: LOOKBACK is excluded from the search — it is fixed at 100 above.
PARAM_GRID = {
    "lstm_units":    [32, 64, 128, 256],    # number of memory units per LSTM layer
    "n_lstm_layers": [1, 2],                # number of stacked LSTM layers
    "dropout_rate":  [0.1, 0.2, 0.3, 0.5], # fraction of units dropped for regularisation
    "learning_rate": [0.0001, 0.001, 0.01], # step size for the Adam optimiser
    "batch_size":    [16, 32, 64],          # samples per gradient update during training
}

# ── Random search budget ───────────────────────────────────────────────────────
# Controls how many combinations to sample from the full pool.
# Rule of thumb: 20–60 often finds near-optimal results.
N_RANDOM_DRAWS = 60







# ── Horizon-tagged filename helper ────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function creates a short label based on the selected forecast horizon.
# For example, t+1 becomes "t1", t+7 becomes "t7", and t+30 becomes "t30".
# The tag is used in output filenames for datasets, models, scalers,
# and feature column lists.
#
# This keeps outputs from different forecast horizons clearly separated.
# It also prevents files from being overwritten when the same script is reused
# for multiple forecasting horizons.
def horizon_tag() -> str:
    """Return a short suffix string that reflects the current forecast horizon.
    Examples: FORECAST_HORIZON=1  → 't1'
              FORECAST_HORIZON=7  → 't7'
              FORECAST_HORIZON=30 → 't30'
    """
    return f"t{FORECAST_HORIZON}"








# ── Data loading & feature engineering ───────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function creates the feature columns used by the LSTM.
# Categorical variables are converted into numeric codes where needed.
# Lagged log returns are added so the model can use recent return history.
# A rolling volatility feature is calculated using previous returns only.
#
# All other numeric columns are shifted by one period to avoid look-ahead bias.
# Target columns for all horizons are excluded from feature construction.
# Rows with NaN values caused by shifting are removed.
# The returned dataframe is ready for sequence construction.
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix from the raw DataFrame.

    All features are lagged by at least one period so that only information
    available strictly before t+H is used.  This prevents data leakage.

    Categorical columns (e.g. Market_Regime) are label-encoded before
    being used as features.

    Features added:
      lag_1 ... lag_5  : log return at t-1 through t-5
      rolling_vol_5    : standard deviation of log returns over the previous
                         5 days (itself shifted by 1 to avoid leakage)
      All other numeric columns are included with a 1-period lag (_lag1).
    """
    df = df.copy()

    # Step 1: Label-encode any non-numeric columns (except the date and asset columns)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_meta = [c for c in cat_cols if c not in (DATE_COLUMN, "asset")]
    for col in non_meta:
        df[col] = pd.Categorical(df[col]).codes

    # Step 2: Build all new feature columns in a dictionary first,
    # then add them to the DataFrame in one go (avoids fragmentation warnings).
    new_cols = {}

    # Lagged log returns: lag_1 is yesterday's return, lag_2 is two days ago, etc.
    for lag in LAG_FEATURES:
        new_cols[f"lag_{lag}"] = df["log_return"].shift(lag)

    # Rolling volatility: std of log returns over the last 5 days,
    # shifted by 1 so it does not include today's value.
    new_cols["rolling_vol_5"] = (
        df["log_return"].shift(1).rolling(window=ROLLING_WINDOW).std()
    )

    # Lag all other numeric columns by 1 period to avoid look-ahead bias.
    # NOTE: We skip target columns for ALL possible horizons (not just the
    # current one) so that the feature set remains consistent across horizon
    # runs and no future target information leaks into the features.
    skip_cols = (
        {"log_return"}
        | {f"lag_{l}" for l in LAG_FEATURES}
        | {"rolling_vol_5"}
    )
    # Exclude any column whose name starts with "target_" to cover all horizons
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    extra_cols   = [
        c for c in numeric_cols
        if c not in skip_cols and not c.startswith("target_")
    ]

    for col in extra_cols:
        new_cols[f"{col}_lag1"] = df[col].shift(1)

    # Step 3: Concatenate all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Step 4: Drop rows with any NaN values introduced by shifting
    df = df.dropna().reset_index(drop=True)

    return df










# Load Bitcoin data and create the t+H prediction target      (IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function reads the raw Bitcoin CSV file.
# It sorts rows chronologically and checks that the close price column exists.
# The close price is converted to numeric format and invalid rows are removed.
#
# Daily log returns are calculated from the close price.
# The target is created by shifting the daily log return by FORECAST_HORIZON.
# For this file, FORECAST_HORIZON = 1, so the model predicts t+1 log return.
# Feature engineering is then applied to create the final input variables.
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file, compute log returns and the t+H target,
    then apply feature engineering.

    V2 change: target column name and shift(-FORECAST_HORIZON) are driven
    by the module-level FORECAST_HORIZON constant, so changing that value
    automatically propagates to the target and all downstream steps.
    """
    logging.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sort chronologically if a date column exists
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Validate that the price column is present
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Column '{PRICE_COLUMN}' not found in {csv_path}")

    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
    df = df.dropna(subset=[PRICE_COLUMN]).reset_index(drop=True)

    # Daily log return: r_t = log(P_t / P_{t-1})
    df["log_return"] = np.log(df[PRICE_COLUMN]).diff()

    # The t+H target is the log return H periods ahead of each row.
    # FORECAST_HORIZON drives the shift so the script is horizon-agnostic.
    target_col = f"target_log_return_t_plus_{FORECAST_HORIZON}"
    df[target_col] = df["log_return"].shift(-FORECAST_HORIZON)

    # Drop rows where either the return or the target is NaN
    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    # Apply feature engineering (lags, rolling vol, lagged numeric columns)
    df = build_features(df)

    logging.info(
        "Loaded %d rows with %d features after engineering (horizon t+%d)",
        len(df), len(get_feature_cols(df)), FORECAST_HORIZON,
    )
    return df












# Select the columns used as LSTM input features           (IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function returns the ordered list of model input columns.
# It excludes the date, close price, raw log return, asset column,
# and every column that starts with "target_".
#
# Excluding all target columns avoids leakage when the same script is reused
# for different forecast horizons.
# The resulting feature list is used consistently for scaling, sequence
# construction, training, validation, testing, and saved feature lists.
def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return the ordered list of feature columns used as model inputs.
    Excludes the target column, the date column, the raw price column,
    the raw log return column, and the 'asset' column if present.

    V2 change: excludes ALL columns that start with 'target_' rather than
    only the current-horizon target, so the feature set is identical
    regardless of which horizon is being run.
    """
    exclude = {
        DATE_COLUMN,
        PRICE_COLUMN,
        "log_return",
        "asset",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith("target_")
    ]
    return feature_cols










# ── Train / validation / test split ───────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function divides the processed dataframe into chronological subsets.
# The split follows the predefined 70/15/15 ratio.
# The training set is used for model fitting.
# The validation set is used for random search hyperparameter selection.
# The test set is reserved for final out-of-sample evaluation.
#
# No shuffling is applied because this is a time-series forecasting task.
# This preserves the correct past-to-future structure.
def split_data(df: pd.DataFrame):
    """
    Split the DataFrame chronologically into train, validation, and test sets
    using the 70 / 15 / 15 ratio.  No shuffling — temporal order is preserved.
    """
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test  = df.iloc[valid_end:].copy()

    logging.info(
        "Split (horizon t+%d) — Train: %d  |  Valid: %d  |  Test: %d",
        FORECAST_HORIZON, len(train), len(valid), len(test),
    )
    return train, valid, test











# ── NEW (V2): Dataset saving ──────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function exports the processed splits to horizon-tagged CSV files.
# Only the date column, selected feature columns, and target column are saved.
# The data is saved before scaling, so future analysis can reuse raw features.
#
# This supports reproducibility and later SHAP or permutation importance analysis.
# The function also saves the ordered feature column list to a text file.
# File names include the horizon tag so multiple horizon runs stay separated.
def save_split_datasets(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list,
    target_col:   str,
    output_dir:   str = ".",
) -> None:
    """
    Save the three unscaled splits to CSV files for reproducible reuse.

    IMPORTANT — What is saved and why
    ───────────────────────────────────
    • Only the feature columns and the target column are saved.
      This produces a clean, minimal dataset that exactly matches what the
      model sees (before scaling), with no superfluous columns.
    • Scaling is NOT applied here.  The saved data is raw so that the scaler
      can be re-fit on training data in any future analysis, maintaining
      correct train-only scaler fitting and avoiding leakage.
    • The date column is included (if present) to preserve temporal context
      for later analysis and visualisation.
    • Files are named with the horizon tag so that multiple runs (t+1, t+7,
      t+30) coexist in the same directory without overwriting each other.

    SHAP / permutation importance workflow (downstream)
    ────────────────────────────────────────────────────
    Load test_tH.csv, apply scaler_tv_tH.joblib, build sequences, then call
    your SHAP explainer or permutation importance routine on the test set.
    No retraining is needed — load lstm_model_tH.keras directly.

    Parameters
    ----------
    train_df, valid_df, test_df : the three unscaled split DataFrames
    feature_cols : ordered list of feature column names
    target_col   : name of the target column
    output_dir   : directory where the CSV files are written (default: cwd)
    """
    tag     = horizon_tag()
    out     = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Columns to save: date (if present) + features + target
    save_cols = []
    if DATE_COLUMN in train_df.columns:
        save_cols.append(DATE_COLUMN)
    save_cols.extend(feature_cols)
    save_cols.append(target_col)

    for name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        # Select only the desired columns (handle the case where date may be absent)
        cols_present = [c for c in save_cols if c in split_df.columns]
        filepath     = out / f"{name}_{tag}.csv"
        split_df[cols_present].to_csv(filepath, index=False)
        logging.info(
            "Saved %s split (%d rows, %d cols) → %s",
            name, len(split_df), len(cols_present), filepath,
        )

    # Save the ordered feature column list to a text file for easy reconstruction
    feat_file = out / f"feature_cols_{tag}.txt"
    feat_file.write_text("\n".join(feature_cols))
    logging.info("Saved feature column list (%d cols) → %s", len(feature_cols), feat_file)









# ── NEW (V2): Model and scaler persistence ───────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function saves the final LSTM model and StandardScaler to disk.
# The model is saved as a Keras file containing architecture and weights.
# The scaler is saved with joblib.
#
# Both artefacts correspond to the final train+validation setup used before
# evaluating the test set.
# Saving them allows later SHAP or permutation importance analysis without
# retraining the model or refitting the scaler.
def save_model_and_scaler(
    model:      Sequential,
    scaler:     StandardScaler,
    output_dir: str = ".",
) -> None:
    """
    Persist the trained LSTM model and the fitted scaler for later reuse.

    Files written
    ─────────────
    lstm_model_t{H}.keras       — full Keras model (weights + architecture)
    scaler_tv_t{H}.joblib       — StandardScaler fitted on train+validation

    Both artefacts correspond to the final model used for test evaluation.
    The scaler was fit on train + validation features only, consistent with
    the methodology used during evaluation.

    Why save these?
    ───────────────
    SHAP values and permutation importance require the trained model and the
    exact same scaler that was used during training.  Saving them here avoids
    any risk of fitting a different scaler or retraining the model later.

    Loading example (in a separate analysis script)
    ───────────────────────────────────────────────
        import joblib, tensorflow as tf

        model  = tf.keras.models.load_model("lstm_model_t1.keras")
        scaler = joblib.load("scaler_tv_t1.joblib")
        test   = pd.read_csv("test_t1.csv")
        feature_cols = open("feature_cols_t1.txt").read().splitlines()

        X_test_scaled = scaler.transform(test[feature_cols].values.astype("float32"))
        # → build sequences and run SHAP / permutation importance
    """
    tag     = horizon_tag()
    out     = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path  = out / f"lstm_model_{tag}.keras"
    scaler_path = out / f"scaler_tv_{tag}.joblib"

    model.save(str(model_path))
    logging.info("Saved trained LSTM model → %s", model_path)

    joblib.dump(scaler, str(scaler_path))
    logging.info("Saved fitted scaler → %s", scaler_path)









# ── Metrics ─────────────────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function computes Mean Absolute Percentage Error.
# Bitcoin log returns can be very close to zero.
# Direct division by these values can create unstable or extreme errors.
#
# A small epsilon value is therefore used as a minimum denominator.
# This avoids division by zero and makes the MAPE calculation more stable.
# The result is returned as a percentage.
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error, guarded against near-zero actual values.
    Near-zero log returns are common and would cause division by zero,
    so a small epsilon is used in the denominator.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom  = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)






# Calculate directional accuracy                 (IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function measures whether the model predicts the correct return direction.
# It compares the sign of the predicted log return with the sign of the actual
# future log return.
#
# A prediction is counted as correct when both signs match.
# The result is returned as a percentage of correctly predicted directions.
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of predictions where the model correctly predicted
    whether the return would be positive or negative.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0)





# Calculate model evaluation metrics               (IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function computes the main evaluation metrics for the LSTM forecasts.
# MAE measures the average absolute forecasting error.
# RMSE gives stronger weight to larger prediction errors.
# MAPE gives a percentage-based error estimate using the safe MAPE helper.
# Directional Accuracy checks whether the predicted sign is correct.
#
# The metrics are returned in a dictionary for logging, printing, saving,
# and comparison with other models.
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all four evaluation metrics and return them as a dictionary."""
    return {
        "MAE":                  mean_absolute_error(y_true, y_pred),
        "RMSE":                 sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE":                 safe_mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }









# ── Sequence construction ────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function transforms a 2-D feature matrix into 3-D sequences.
# Each sequence contains the previous LOOKBACK observations.
# With LOOKBACK = 100, every prediction uses the last 100 days of inputs.
#
# The target for each sequence is the value directly after the input window.
# This ensures the model only learns from historical information.
# If there are not enough rows to build a sequence, empty arrays are returned.
# The output shape is suitable for Keras LSTM input.
def build_sequences(X: np.ndarray, y: np.ndarray):
    """
    Convert a 2-D scaled feature array into overlapping 3-D LSTM sequences
    using the fixed LOOKBACK window.

    For each index i >= LOOKBACK the input sequence spans rows
    [i - LOOKBACK : i], and the target is y[i].  Only past information
    is used; no future rows enter any sequence.

    Parameters
    ----------
    X : shape (T, F) — scaled feature array
    y : shape (T,)   — target array

    Returns
    -------
    X_seq : shape (N, LOOKBACK, F)
    y_seq : shape (N,)
    """
    n = len(X)

    # Not enough rows to form even one sequence
    if n <= LOOKBACK:
        empty_X = np.empty((0, LOOKBACK, X.shape[1]), dtype=np.float32)
        empty_y = np.empty(0, dtype=np.float32)
        return empty_X, empty_y

    # Number of valid sequences
    N = n - LOOKBACK

    # Pre-allocate arrays for efficiency
    X_seq = np.empty((N, LOOKBACK, X.shape[1]), dtype=np.float32)
    y_seq = np.empty(N, dtype=np.float32)

    # Build each sequence one by one (explicit loop for clarity)
    for k in range(N):
        # The input window is rows k to k+LOOKBACK (exclusive)
        X_seq[k] = X[k : k + LOOKBACK]
        # The target is the value right after the window
        y_seq[k] = y[k + LOOKBACK]

    return X_seq, y_seq







# ── LSTM model builder ─────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function defines the LSTM model architecture.
# It creates an input layer based on LOOKBACK and the number of features.
# Depending on the parameters, it builds either one LSTM layer or two stacked
# LSTM layers.
#
# A dropout layer is added for regularisation.
# A single dense linear output unit is used because this is a regression task.
# The model is compiled with the Adam optimiser and MSE loss.
# The compiled Keras model is returned for training.
def build_lstm(
    n_features:    int,
    lstm_units:    int,
    n_lstm_layers: int,
    dropout_rate:  float,
    learning_rate: float,
) -> Sequential:
    """
    Build and compile an LSTM Sequential model.

    Architecture:
      Input(LOOKBACK, n_features)
      → LSTM(lstm_units) [× 1 or 2 stacked layers]
      → Dropout(dropout_rate)
      → Dense(1, linear activation)

    The linear output activation is appropriate for regression.
    MSE is used as the loss function, consistent with a regression objective.
    """
    model = Sequential()

    # Explicit Input layer (recommended by Keras 3.x)
    model.add(Input(shape=(LOOKBACK, n_features)))

    if n_lstm_layers == 1:
        # Single LSTM layer — no need to return sequences
        model.add(LSTM(lstm_units))
    else:
        # Two stacked LSTM layers.
        # The first must return sequences so the second can process them.
        model.add(LSTM(lstm_units, return_sequences=True))
        model.add(LSTM(lstm_units))

    # Dropout for regularisation (reduces overfitting)
    model.add(Dropout(dropout_rate))

    # Single linear output unit (regression)
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model








# ── Initial training (with early stopping) ────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function builds and trains the LSTM using the supplied hyperparameters.
# It trains on pre-built LOOKBACK sequences.
# If enough sequences are available, the last EARLY_STOP_WINDOW sequences are
# held out as a small validation set for early stopping.
#
# Early stopping stops training when validation loss no longer improves,
# which helps reduce overfitting during the initial training phase.
# If too few sequences exist, the model trains on all available sequences.
# The trained model is returned for later online or validation evaluation.
def initial_train(
    X_seq:  np.ndarray,
    y_seq:  np.ndarray,
    params: dict,
) -> Sequential:
    """
    Build and perform the initial full training of the LSTM on a sequence
    dataset, with early stopping on a held-out trailing validation window.

    The trailing EARLY_STOP_WINDOW sequences are held out as the early-stop
    validation set.  This prevents overfitting during the offline training phase
    before online updates begin.

    Parameters
    ----------
    X_seq  : (N, LOOKBACK, F) sequences built from the training pool
    y_seq  : (N,) corresponding targets
    params : hyperparameter dictionary (must include lstm_units, n_lstm_layers,
             dropout_rate, learning_rate, batch_size)

    Returns
    -------
    Trained Keras Sequential model.
    """
    n_features = X_seq.shape[2]

    # Build the model using the provided hyperparameters
    model = build_lstm(
        n_features    = n_features,
        lstm_units    = params["lstm_units"],
        n_lstm_layers = params["n_lstm_layers"],
        dropout_rate  = params["dropout_rate"],
        learning_rate = params["learning_rate"],
    )

    # If there are not enough sequences for an early-stop split,
    # train on all available sequences without validation monitoring.
    if len(X_seq) <= EARLY_STOP_WINDOW:
        model.fit(
            X_seq, y_seq,
            epochs     = MAX_EPOCHS,
            batch_size = params["batch_size"],
            verbose    = 0,
        )
        return model

    # Split sequences: most go to training, the last EARLY_STOP_WINDOW to validation
    X_fit = X_seq[:-EARLY_STOP_WINDOW]
    y_fit = y_seq[:-EARLY_STOP_WINDOW]
    X_val = X_seq[-EARLY_STOP_WINDOW:]
    y_val = y_seq[-EARLY_STOP_WINDOW:]

    # Early stopping: stop if validation loss does not improve for EARLY_STOP_ROUNDS epochs
    early_stop = EarlyStopping(
        monitor              = "val_loss",
        patience             = EARLY_STOP_ROUNDS,
        restore_best_weights = True,
        verbose              = 0,
    )

    # Suppress Keras warnings during training (e.g. about small batch sizes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_fit, y_fit,
            validation_data = (X_val, y_val),
            epochs          = MAX_EPOCHS,
            batch_size      = params["batch_size"],
            callbacks       = [early_stop],
            verbose         = 0,
        )
    return model









# ── Online evaluation loop ─────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function simulates a streaming forecasting setup.
# For each new evaluation row, the model first predicts using the latest
# LOOKBACK rows from the context buffer.
# The realised target is then used to update the model with train_on_batch.
#
# After each step, the new feature row is added to the context buffer.
# This means the model gradually adapts as new observations arrive.
# This function is intended for final online train/test evaluation.
def online_evaluate(
    model:            Sequential,
    X_eval_scaled:    np.ndarray,
    y_eval:           np.ndarray,
    batch_size:       int,
    X_context_scaled: np.ndarray,
):
    """
    Online predict-then-update loop over an evaluation block.

    For each observation i in the evaluation set:
      1. Construct the prediction sequence from the last LOOKBACK rows of
         the combined context (training history + evaluation rows seen so far).
      2. Predict t+H log return.
      3. Update model weights via train_on_batch on the new (sequence, target).
      4. Absorb the new row into the rolling context buffer.

    This simulates a streaming environment where the model adapts after each
    new data point arrives.

    Parameters
    ----------
    model             : pre-trained Keras LSTM model
    X_eval_scaled     : (M, F) scaled features for the evaluation block
    y_eval            : (M,) targets for the evaluation block
    batch_size        : mini-batch size for train_on_batch
    X_context_scaled  : (T, F) scaled features from the training history
                        preceding the evaluation block (at least LOOKBACK rows)

    Returns
    -------
    predictions : numpy array of predicted log returns
    actuals     : numpy array of actual log returns
    """
    predictions = []   # Will hold one float per evaluated observation
    actuals     = []   # Will hold the corresponding actual return

    # The context buffer starts with the training history.
    # It grows by one row after each step as new observations are absorbed.
    # Only the last LOOKBACK rows are ever needed for a prediction.
    context = X_context_scaled.copy()

    for i in range(len(X_eval_scaled)):

        # If we do not have enough history yet, skip prediction and just absorb the row
        if len(context) < LOOKBACK:
            new_row = X_eval_scaled[[i]]                      # shape (1, F)
            context = np.vstack([context, new_row])
            continue

        # Step 1: Build the prediction sequence from the last LOOKBACK rows
        x_seq = context[-LOOKBACK:][np.newaxis, :, :].astype(np.float32)  # (1, LOOKBACK, F)

        # Step 2: Predict the t+H log return
        pred  = float(model.predict(x_seq, verbose=0)[0, 0])
        predictions.append(pred)
        actuals.append(float(y_eval[i]))

        # Step 3: Online update — adjust model weights using the newly revealed actual
        y_new = np.array([[y_eval[i]]], dtype=np.float32)     # shape (1, 1)
        model.train_on_batch(x_seq, y_new)

        # Step 4: Absorb the new row into the context buffer
        new_row = X_eval_scaled[[i]]                          # shape (1, F)
        context = np.vstack([context, new_row])

    predictions_array = np.asarray(predictions, dtype=float)
    actuals_array     = np.asarray(actuals,     dtype=float)

    return predictions_array, actuals_array









# ── Leakage-free validation evaluation loop ─────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function is similar to online_evaluate, but deliberately does not call
# train_on_batch after each prediction.
# It is used during random search validation to avoid data leakage.
#
# The model predicts sequentially using the rolling context buffer.
# New validation feature rows are added to the context, but validation targets
# are never used to update the model weights.
# This keeps hyperparameter selection based on genuinely unseen validation data.
def predict_only_evaluate(
    model:            Sequential,
    X_eval_scaled:    np.ndarray,
    y_eval:           np.ndarray,
    X_context_scaled: np.ndarray,
):
    """
    Online-style prediction loop over an evaluation block with NO weight updates.

    Used exclusively during random search (hyperparameter optimisation) to
    evaluate each candidate model on the validation set without leaking
    validation targets into the model weights.

    For each observation i in the evaluation set:
      1. Construct the prediction sequence from the last LOOKBACK rows of
         the combined context (training history + validation rows seen so far).
      2. Predict t+H log return.
      3. Absorb the new feature row into the rolling context buffer.
         NOTE: train_on_batch is intentionally omitted — model weights are
         NOT updated during validation, keeping evaluation purely predictive.

    Contrast with online_evaluate (used on the test set), which performs a
    train_on_batch step after each prediction to simulate true online learning.

    Parameters
    ----------
    model             : pre-trained Keras LSTM model
    X_eval_scaled     : (M, F) scaled features for the evaluation block
    y_eval            : (M,) targets for the evaluation block
    X_context_scaled  : (T, F) scaled features from the training history
                        preceding the evaluation block (at least LOOKBACK rows)

    Returns
    -------
    predictions : numpy array of predicted log returns
    actuals     : numpy array of actual log returns
    """
    predictions = []   # Will hold one float per evaluated observation
    actuals     = []   # Will hold the corresponding actual return

    # The context buffer starts with the training history.
    # It grows by one row after each step as new feature rows are absorbed.
    # Only the last LOOKBACK rows are ever needed for a prediction.
    context = X_context_scaled.copy()

    for i in range(len(X_eval_scaled)):

        # If we do not have enough history yet, skip prediction and just absorb the row
        if len(context) < LOOKBACK:
            new_row = X_eval_scaled[[i]]                      # shape (1, F)
            context = np.vstack([context, new_row])
            continue

        # Step 1: Build the prediction sequence from the last LOOKBACK rows
        x_seq = context[-LOOKBACK:][np.newaxis, :, :].astype(np.float32)  # (1, LOOKBACK, F)

        # Step 2: Predict the t+H log return
        pred  = float(model.predict(x_seq, verbose=0)[0, 0])
        predictions.append(pred)
        actuals.append(float(y_eval[i]))

        # Step 3: Absorb the new feature row into the context buffer.
        # train_on_batch is deliberately NOT called here — validation targets
        # must not influence model weights during hyperparameter search.
        new_row = X_eval_scaled[[i]]                          # shape (1, F)
        context = np.vstack([context, new_row])

    predictions_array = np.asarray(predictions, dtype=float)
    actuals_array     = np.asarray(actuals,     dtype=float)

    return predictions_array, actuals_array






# ── Scaler helper ────────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function fits a StandardScaler on the provided feature matrix.
# It returns both the fitted scaler and the scaled data.
#
# The scaler should only be fitted on training data or train+validation data.
# It must not be fitted on validation or test data during evaluation,
# because that would introduce data leakage.
# The fitted scaler is later reused to transform unseen data.
def fit_scaler(X: np.ndarray):
    """
    Fit a StandardScaler on X and return the fitted scaler and the scaled array.

    IMPORTANT: The scaler must always be fit on training data only.
    Fitting on validation or test data would introduce data leakage.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    return scaler, X_scaled







# ── Hyperparameter search helpers ─────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function creates the full search space from PARAM_GRID.
# It constructs every possible combination of LSTM units, layer count,
# dropout rate, learning rate, and batch size.
#
# The output is a list of dictionaries, where each dictionary represents
# one complete hyperparameter configuration.
# This list is later shuffled and sampled for random search.
# The function is written explicitly for readability and transparency.
def build_all_combinations(param_grid: dict) -> list:
    """
    Build a list of all possible hyperparameter combinations from param_grid.

    This is the Cartesian product of all value lists in the dictionary.
    For example, if param_grid = {"a": [1, 2], "b": [10, 20]}, the result is:
      [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]

    Written as explicit nested loops for clarity rather than a one-liner.
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    # Start with a list containing one empty combination
    all_combinations = [{}]

    # For each hyperparameter, extend every existing combination with each possible value
    for key, value_list in zip(keys, values):
        new_combinations = []
        for existing_combo in all_combinations:
            for value in value_list:
                # Copy the existing combination and add the new key-value pair
                new_combo = dict(existing_combo)
                new_combo[key] = value
                new_combinations.append(new_combo)
        all_combinations = new_combinations

    return all_combinations






# ── Random search ──────────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function samples a subset of hyperparameter combinations from PARAM_GRID.
# Each sampled configuration is trained on the training split.
# Validation performance is measured using predict_only_evaluate.
#
# This is important because validation targets are not used for model updates,
# preventing leakage during hyperparameter selection.
# Metrics are stored for every successful configuration.
# The configuration with the lowest validation RMSE is selected as best.
# The function returns both the best parameters and the full results table.
def random_search(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    feature_cols: list,
    target_col:   str,
):
    """
    Randomly sample N_RANDOM_DRAWS combinations from PARAM_GRID and evaluate
    each one on the validation set.  The combination with the lowest validation
    RMSE is selected as best.

    Steps:
    1. Build the full pool of all possible combinations (288 total).
    2. Shuffle the pool randomly using RANDOM_SEED for reproducibility.
    3. Take only the first N_RANDOM_DRAWS combinations from the shuffled pool.
    4. Evaluate those N_RANDOM_DRAWS combinations (same procedure as grid search).

    Everything else is identical: scaler pre-computation, initial_train,
    online_evaluate, metric calculation, and best-parameter selection.

    Parameters
    ----------
    train_df     : training split DataFrame
    valid_df     : validation split DataFrame
    feature_cols : list of input feature column names
    target_col   : name of the target column

    Returns
    -------
    best_params  : dict with the best hyperparameter values
    results_df   : DataFrame with all N_RANDOM_DRAWS candidate results,
                   sorted by validation RMSE
    """
    # Step 1: Build all 288 possible combinations
    all_combinations   = build_all_combinations(PARAM_GRID)
    total_combinations = len(all_combinations)

    # Step 2: Shuffle the list of combinations randomly (seeded for reproducibility)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_combinations)

    # Step 3: Take only the first N_RANDOM_DRAWS combinations from the shuffled list
    n_draws              = min(N_RANDOM_DRAWS, total_combinations)
    sampled_combinations = all_combinations[:n_draws]

    logging.info(
        "Random search (horizon t+%d): sampling %d / %d combinations "
        "| lookback=%d (fixed) | seed=%d",
        FORECAST_HORIZON, n_draws, total_combinations, LOOKBACK, RANDOM_SEED,
    )

    # Pre-compute scaled training data once — reused for every candidate
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train     = train_df[target_col].values.astype(np.float32)
    scaler, X_train_scaled = fit_scaler(X_train_raw)

    # Scale the validation features using the training scaler (no re-fitting)
    X_valid_raw    = valid_df[feature_cols].values.astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid_raw).astype(np.float32)
    y_valid        = valid_df[target_col].values.astype(np.float32)

    # Build training sequences once — the same sequences are used for all candidates
    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train)
    if len(X_train_seq) == 0:
        raise RuntimeError(
            f"Training data too short for LOOKBACK={LOOKBACK}. "
            f"Need at least {LOOKBACK + 1} rows."
        )

    # Evaluate each sampled combination one by one
    results = []
    for combo_number, params in enumerate(sampled_combinations, start=1):
        try:
            logging.info(
                "Random search: evaluating combination %d / %d  params=%s",
                combo_number, n_draws, params,
            )

            # Train the model on the training sequences
            model = initial_train(X_train_seq, y_train_seq, params)

            # Run prediction-only loop over the validation block.
            # predict_only_evaluate is used here (not online_evaluate) to
            # prevent validation targets from updating model weights, which
            # would constitute data leakage during hyperparameter search.
            pred, actual = predict_only_evaluate(
                model            = model,
                X_eval_scaled    = X_valid_scaled,
                y_eval           = y_valid,
                X_context_scaled = X_train_scaled,
            )

            # Skip this candidate if no predictions were produced
            if len(pred) == 0:
                logging.warning("params=%s produced no predictions — skipping.", params)
                continue

            # Compute validation metrics for this candidate
            metrics = calculate_metrics(actual, pred)

            # Store the hyperparameters and their validation metrics together
            row = {
                **params,
                "validation_rmse":                 metrics["RMSE"],
                "validation_mae":                  metrics["MAE"],
                "validation_mape":                 metrics["MAPE"],
                "validation_directional_accuracy": metrics["Directional_Accuracy"],
            }
            results.append(row)

            logging.info(
                "params=%s | valid RMSE=%.8f | MAE=%.8f | MAPE=%.4f | DA=%.2f%%",
                params, metrics["RMSE"], metrics["MAE"],
                metrics["MAPE"], metrics["Directional_Accuracy"],
            )

        except Exception as exc:
            logging.warning("params=%s failed: %s", params, exc)

    if not results:
        raise RuntimeError("No hyperparameter configuration completed successfully.")

    # Sort all evaluated candidates by validation RMSE — the best one is at the top
    results_df = pd.DataFrame(results).sort_values("validation_rmse").reset_index(drop=True)

    # Extract the best parameter set from the top row
    param_keys  = list(PARAM_GRID.keys())
    best_params = {k: results_df.iloc[0][k] for k in param_keys}

    # Cast integer hyperparameters back to int (pandas may have read them as float)
    for key in ("lstm_units", "n_lstm_layers", "batch_size"):
        best_params[key] = int(best_params[key])

    logging.info(
        "Best params: %s  |  validation RMSE: %.8f",
        best_params, results_df.iloc[0]["validation_rmse"],
    )
    return best_params, results_df








# ── Model evaluation (train split and test split) ────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function evaluates the selected hyperparameter configuration.
# For the training split, a warm-up section is used to initialise the model.
# The remaining training rows are evaluated using online predict-then-update.
#
# For the test split, the model is retrained on train+validation data.
# The scaler is fitted only on train+validation features.
# The test set is then evaluated using online learning.
# The final trained model and scaler are also returned for saving.
def fit_and_evaluate(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    params:       dict,
    feature_cols: list,
    target_col:   str,
):
    """
    Evaluate the best hyperparameter configuration on both the training
    split (in-sample) and the held-out test set (out-of-sample).

    IN-SAMPLE EVALUATION (training split)
    ──────────────────────────────────────
    The model is trained on the first (LOOKBACK + EARLY_STOP_WINDOW + 1)
    rows as the warm-up, then the online loop runs over the remainder of
    the training split, updating weights after each observation.
    This gives an in-sample measure of fit.

    OUT-OF-SAMPLE EVALUATION (test split)
    ───────────────────────────────────────
    The scaler is re-fit on train + validation features.  The model is
    retrained from scratch on all sequences from train + validation data.
    The online loop then runs over the test split: predict → train_on_batch.

    V2 change: the final model and scaler (fitted on train+validation) are
    returned so that the caller can persist them for later SHAP analysis.

    Returns
    -------
    Tuple of:
      (test_pred, test_actual, test_metrics,
       train_pred, train_actual, train_metrics,
       final_model, scaler_tv)

    The last two items (final_model, scaler_tv) are NEW in V2 and are used
    by save_model_and_scaler() to write artefacts to disk.
    """

    # ── In-sample: online loop over the training split ─────────────────────────
    warmup = LOOKBACK + EARLY_STOP_WINDOW + 1

    train_init = train_df.iloc[:warmup]   # Used to initialise the model
    train_eval = train_df.iloc[warmup:]   # Evaluated in the online loop

    # Fit the scaler on the warm-up portion only (no leakage)
    X_init_raw = train_init[feature_cols].values.astype(np.float32)
    y_init     = train_init[target_col].values.astype(np.float32)
    scaler_train, X_init_scaled = fit_scaler(X_init_raw)

    # Scale the rest of the training set using the same scaler (no re-fitting)
    X_train_eval_raw    = train_eval[feature_cols].values.astype(np.float32)
    X_train_eval_scaled = scaler_train.transform(X_train_eval_raw)
    y_train_eval        = train_eval[target_col].values.astype(np.float32)

    # Build sequences from the warm-up data and do the initial offline training
    X_init_seq, y_init_seq = build_sequences(X_init_scaled, y_init)
    train_model = initial_train(X_init_seq, y_init_seq, params)

    # Run the online loop over the remaining training rows
    train_pred, train_actual = online_evaluate(
        model             = train_model,
        X_eval_scaled     = X_train_eval_scaled,
        y_eval            = y_train_eval,
        batch_size        = params["batch_size"],
        X_context_scaled  = X_init_scaled,
    )

    # Compute in-sample metrics
    train_metrics = calculate_metrics(train_actual, train_pred)

    # ── Out-of-sample: retrain on train + validation, then online over test ────
    # Combine train and validation sets for the final model training
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Re-fit the scaler on train + validation (no leakage into the test set)
    X_tv_raw = train_valid_df[feature_cols].values.astype(np.float32)
    y_tv     = train_valid_df[target_col].values.astype(np.float32)
    scaler_tv, X_tv_scaled = fit_scaler(X_tv_raw)

    # Build sequences and train the model on the full train + validation data
    X_tv_seq, y_tv_seq = build_sequences(X_tv_scaled, y_tv)
    test_model = initial_train(X_tv_seq, y_tv_seq, params)

    # Scale the test features using the train+validation scaler (no re-fitting)
    X_test_raw    = test_df[feature_cols].values.astype(np.float32)
    X_test_scaled = scaler_tv.transform(X_test_raw)
    y_test        = test_df[target_col].values.astype(np.float32)

    # Run the online loop over the test set
    test_pred, test_actual = online_evaluate(
        model             = test_model,
        X_eval_scaled     = X_test_scaled,
        y_eval            = y_test,
        batch_size        = params["batch_size"],
        X_context_scaled  = X_tv_scaled,
    )

    # Compute out-of-sample metrics
    test_metrics = calculate_metrics(test_actual, test_pred)

    # V2: return the final model and scaler so they can be saved by the caller
    return (test_pred, test_actual, test_metrics,
            train_pred, train_actual, train_metrics,
            test_model, scaler_tv)








# ── Output ───────────────────────────────────────────────────(IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This function exports the main output files from the after-optimisation run.
# First, it saves all evaluated random search configurations with validation
# metrics, sorted by validation RMSE.
#
# Second, it saves test predictions aligned with the original test rows.
# Actual and predicted log returns are added, along with the best parameters,
# lookback window, forecast horizon, and number of random draws.
# These files can later be used for plotting, reporting, and comparison.
def save_outputs(
    full_df:        pd.DataFrame,
    train_len:      int,
    valid_len:      int,
    test_pred:      np.ndarray,
    test_actual:    np.ndarray,
    best_params:    dict,
    search_results: pd.DataFrame,
    target_col:     str,
) -> None:
    """
    Save two CSV files (unchanged from V1):
      1. lstm_random_search_results_1.csv
         All hyperparameter combinations evaluated during the random search,
         sorted by validation RMSE.
      2. lstm_test_predictions_random_search_1.csv
         The test-set predictions alongside the actual returns and the best
         hyperparameter values used, for downstream analysis.
    """
    # Save the random search results table
    search_results.to_csv("lstm_random_search_results_1.csv", index=False)
    logging.info("Saved → lstm_random_search_results_1.csv")

    # Build the predictions DataFrame from the test portion of the full dataset
    start_idx = train_len + valid_len
    pred_df   = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()

    # Add prediction columns
    pred_df[f"actual_log_return_t_plus_{FORECAST_HORIZON}"]    = test_actual
    pred_df[f"predicted_log_return_t_plus_{FORECAST_HORIZON}"] = test_pred

    # Record the best hyperparameters and lookback for traceability
    pred_df["lookback_window"]   = LOOKBACK
    pred_df["forecast_horizon"]  = FORECAST_HORIZON
    pred_df["n_random_draws"]    = N_RANDOM_DRAWS
    for key, value in best_params.items():
        pred_df[f"best_{key}"] = value

    pred_df.to_csv("lstm_test_predictions_random_search_1.csv", index=False)
    logging.info("Saved → lstm_test_predictions_random_search_1.csv")









# ── Main ───────────────────────────────────────────────────────────────────────
# Run the complete LSTM after-optimisation pipeline       (IMPROVED BY CLAUDE Sonnet 4.6)
# ==============================================================================
# This is the main controller function for the script.
# It loads data, creates features, selects input columns, and splits the data.
# The processed unscaled splits are saved for reproducibility and SHAP analysis.
#
# Random search is then used to select the best hyperparameters.
# The best model is evaluated on train and test data using online learning.
# The final model and scaler are saved for later reuse.
# Results are logged, printed, and exported to CSV files.
def lstm_forecast() -> None:
    """
    Main entry point.  Orchestrates the full pipeline:

      1. Load and preprocess data (log returns + feature engineering)
      2. Split into train / validation / test (70/15/15, chronological)
      3. [NEW V2] Save unscaled splits as horizon-tagged CSV files
      4. Run random search over N_RANDOM_DRAWS combinations from PARAM_GRID
      5. Evaluate the best hyperparameter set on train and test splits
      6. [NEW V2] Save the trained model and fitted scaler
      7. Print results to the terminal
      8. Save predictions and search results to CSV

    V2 additions (steps 3 and 6) are clearly marked below.
    All other steps are identical to the original script.
    """

    # Step 1: Load data, compute log returns, build features
    df           = load_data(CSV_PATH)
    feature_cols = get_feature_cols(df)
    target_col   = f"target_log_return_t_plus_{FORECAST_HORIZON}"

    # Step 2: Chronological split (70 / 15 / 15)
    train_df, valid_df, test_df = split_data(df)

    # ── Step 3 [NEW V2]: Save unscaled datasets for reproducible reuse ────────
    # Datasets are saved AFTER feature engineering and BEFORE any scaling.
    # No transformations are applied after the split — zero leakage guarantee.
    # Files are named with the horizon tag so multiple runs coexist safely.
    logging.info(
        "V2: Saving unscaled split datasets for horizon t+%d ...",
        FORECAST_HORIZON,
    )
    save_split_datasets(
        train_df     = train_df,
        valid_df     = valid_df,
        test_df      = test_df,
        feature_cols = feature_cols,
        target_col   = target_col,
        output_dir   = ".",
    )

    # Step 4: Run random search over N_RANDOM_DRAWS combinations
    best_params, search_results = random_search(
        train_df     = train_df,
        valid_df     = valid_df,
        feature_cols = feature_cols,
        target_col   = target_col,
    )

    # Step 5: Train and evaluate using the best hyperparameters
    # V2: fit_and_evaluate now also returns the final model and scaler_tv
    (test_pred, test_actual, test_metrics,
     train_pred, train_actual, train_metrics,
     final_model, scaler_tv) = fit_and_evaluate(
        train_df     = train_df,
        valid_df     = valid_df,
        test_df      = test_df,
        params       = best_params,
        feature_cols = feature_cols,
        target_col   = target_col,
    )

    # ── Step 6 [NEW V2]: Persist the trained model and scaler ─────────────────
    # Both artefacts correspond exactly to the final evaluation run.
    # Saving them here removes any need to retrain when running SHAP later.
    logging.info(
        "V2: Saving trained model and scaler for horizon t+%d ...",
        FORECAST_HORIZON,
    )
    save_model_and_scaler(
        model      = final_model,
        scaler     = scaler_tv,
        output_dir = ".",
    )

    # Step 7: Log results to the terminal
    logging.info(
        "----- TRAIN RESULTS (online learning, t+%d) -----", FORECAST_HORIZON,
    )
    for metric_name, metric_value in train_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info(
        "----- FINAL TEST RESULTS (online learning, t+%d) -----", FORECAST_HORIZON,
    )
    for metric_name, metric_value in test_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    # Step 7b: Print a formatted results block to the console
    total_combos = len(build_all_combinations(PARAM_GRID))

    print("\n" + "=" * 60)
    print("  [After hyperparameter optimisation — random search]")
    print(f"  TRAIN RESULTS  —  LSTM Online Learning (t+{FORECAST_HORIZON} forecast)")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Lookback window         : {LOOKBACK} (fixed)")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  Random draws evaluated  : {N_RANDOM_DRAWS} / {total_combos}")
    print(f"  MAE                     : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {train_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  [After hyperparameter optimisation — random search]")
    print(f"  FINAL TEST RESULTS  —  LSTM Online Learning (t+{FORECAST_HORIZON} forecast)")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Lookback window         : {LOOKBACK} (fixed)")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  Random draws evaluated  : {N_RANDOM_DRAWS} / {total_combos}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    # V2: print a summary of all saved artefacts for this horizon
    tag = horizon_tag()
    print(f"\n  [V2] Artefacts saved for horizon t+{FORECAST_HORIZON}:")
    print(f"    train_{tag}.csv           — unscaled training set")
    print(f"    valid_{tag}.csv           — unscaled validation set")
    print(f"    test_{tag}.csv            — unscaled test set (for SHAP)")
    print(f"    feature_cols_{tag}.txt    — ordered feature column list")
    print(f"    scaler_tv_{tag}.joblib    — StandardScaler (fit on train+valid)")
    print(f"    lstm_model_{tag}.keras    — trained LSTM model")
    print(f"    lstm_random_search_results_1.csv")
    print(f"    lstm_test_predictions_random_search_1.csv")
    print("=" * 60)

    # Step 8: Save predictions and random search results to CSV
    save_outputs(
        full_df        = df,
        train_len      = len(train_df),
        valid_len      = len(valid_df),
        test_pred      = test_pred,
        test_actual    = test_actual,
        best_params    = best_params,
        search_results = search_results,
        target_col     = target_col,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
    )
    lstm_forecast()
