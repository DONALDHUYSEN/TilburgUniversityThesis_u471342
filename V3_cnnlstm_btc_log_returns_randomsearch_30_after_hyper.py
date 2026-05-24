"""
V3_cnnlstm_btc_log_returns_randomsearch_1_after_hyper.py
──────────────────────────────────────────────────────────
CNN–LSTM model for Bitcoin log-return forecasting.
Online learning variant with hyperparameter optimisation via random search.


1. VALIDATION DATA LEAKAGE FIX (critical)
   During random search, validation is now performed by predict_only_evaluate
   instead of online_evaluate.  The new function generates predictions
   sequentially using the rolling context — identical to online_evaluate —
   but deliberately omits the train_on_batch step.  Validation targets
   therefore never influence model weights during hyperparameter search,
   eliminating the data leakage that was present in V2.

   online_evaluate (with train_on_batch) is still used on the training split
   and the test set inside fit_and_evaluate, where true online learning is
   the intended behaviour.

ALL OTHER FUNCTIONALITY IS UNCHANGED
──────────────────────────────────────────────────
  • Model architecture (Conv1D → MaxPooling1D → LSTM → Dropout → Dense)
  • Feature set, data splits, and scaling logic
  • PARAM_GRID, N_RANDOM_DRAWS, and all random-search mechanics
  • Output formats and file-saving logic

1. Dataset saving after feature engineering + split (per horizon):
   Unscaled tabular datasets are saved as:
     train_t{H}.csv, valid_t{H}.csv, test_t{H}.csv
   where H = FORECAST_HORIZON. These are saved BEFORE any scaling so they
   can be reused for feature importance analysis without data leakage.

2. Multi-horizon compatibility:
   The target column is always built using shift(-FORECAST_HORIZON), and
   the feature engineering + split + save pipeline is parameterised by
   FORECAST_HORIZON so different horizons produce clearly labelled outputs.

3. Feature importance readiness:
   Feature columns are consistent, clearly defined, and saved unscaled.
   The test split matches the model's out-of-sample evaluation window exactly.

4. Correct scaling:
   The scaler is fit only on training data. Validation and test sets are
   transformed using the fitted scaler. ONLY unscaled data is saved to CSV.

5. Lookback-independent datasets:
   Saved CSVs contain raw tabular rows, not sequences. Sequence construction
   (build_sequences) remains inside the modelling pipeline only.

6. Model and scaler persistence:
   After final training the best CNN–LSTM model is saved as a .keras file
   and the fitted scaler is saved via joblib. This allows feature importance
   methods (SHAP, permutation) to be applied later without retraining.

MODEL DESCRIPTION
─────────────────
A CNN–LSTM hybrid network is used for sequence-to-one regression.

The CNN sub-network (Conv1D + MaxPooling1D) acts as a local feature extractor
that compresses temporal patterns within sub-windows of the lookback sequence.
The LSTM sub-network then captures longer-range temporal dependencies in the
compressed representation.

Online learning approach: the model is initially trained on the full training
set and subsequently updated after each new observation arrives using Keras's
train_on_batch.  This simulates a streaming data environment in which the
model adapts continuously to new information.

Architecture:
  - Conv1D layer        : local pattern extraction across the time axis.
  - MaxPooling1D layer  : down-sampling / translation invariance.
  - LSTM layer          : sequential dependency modelling.
  - Dropout layer       : regularisation.
  - Dense(1, linear)    : single regression output.
  - Adam optimiser with mean squared error (MSE) loss.

METHODOLOGY
────────────
Phase 1 — Random search on the training and validation splits:
  60 configurations are drawn randomly from the parameter grid.
  Each candidate is trained on the training data and evaluated on the
  validation set using predict_only_evaluate (predict only, no weight updates).
  The candidate with the lowest validation RMSE is selected as the best.

Phase 2 — Final training on train + validation:
  The best configuration is used to retrain the model from scratch on the
  combined train + validation data.  The scaler is re-fit on this combined
  data only (no leakage into the test set).

Phase 3 — Online evaluation on the test split:
  For each observation in the test split:
    1. Predict t+H log return using the last lookback_window scaled rows.
    2. Record the prediction and the realised return.
    3. Update the model weights via train_on_batch on the new observation.

In-sample (training split) results are also produced so they can be compared
directly with the test results and with the baseline script.

KEY DIFFERENCES FROM cnnlstm_btc_log_returns_before_hyper_1.py
────────────────────────────────────────────────────────────────
  - A random search over 60 configurations replaces the single fixed config.
  - PARAM_GRID defines the search space; N_RANDOM_DRAWS = 60 controls iterations.
  - lookback_window is now a tunable hyperparameter (was a fixed global constant).
  - build_sequences() now accepts a lookback argument instead of using LOOKBACK.
  - online_evaluate() now accepts a lookback argument for the same reason.
  - predict_only_evaluate() is introduced for leakage-free validation scoring.
  - A random_search() function handles the full search procedure.
  - Random search results are saved to a separate CSV file.
  - Output CSV files are renamed to reflect the "after" optimisation status.
  - All other logic is preserved without modification.
"""

import logging
import os
import random
import warnings
from math import sqrt

# Silence TensorFlow noise BEFORE importing TensorFlow 
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam


# Reproducibility 
# Fix all random seeds so results are reproducible across runs.
# TensorFlow operations that use internal C++ kernels may still exhibit small
# non-determinism on GPU; setting the environment variable below eliminates
# the remaining sources of non-determinism on CPU.
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"   # Enforce deterministic TF ops (CPU)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# Configuration 
CSV_PATH     = "btc_clean.csv"   # Path to the input data file
PRICE_COLUMN = "Close"           # Name of the price column in the CSV
DATE_COLUMN  = "Date"            # Name of the date column in the CSV

# Train / validation / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

FORECAST_HORIZON  = 30    # Predict the log return H days ahead (t+H)
EARLY_STOP_ROUNDS = 10   # EarlyStopping patience in epochs
EARLY_STOP_WINDOW = 30   # Number of trailing sequences held out for early stopping
MAX_EPOCHS        = 100  # Maximum number of training epochs

LAG_FEATURES   = list(range(1, 6))  # Lagged log returns: t-1 to t-5
ROLLING_WINDOW = 5                  # Window size for rolling volatility feature

# Output directory for datasets, model, and scaler artefacts.
# Use "." (current directory) or set to a sub-folder such as "outputs".
OUTPUT_DIR = "."


# Random search settings
# N_RANDOM_DRAWS controls how many configurations are tried during the search.
# Each draw samples one value from each list in PARAM_GRID independently
# and at random (uniform sampling without replacement within each draw).
N_RANDOM_DRAWS = 60

# PARAM_GRID defines the candidate values for each hyperparameter.
# lookback_window is included here because it is now tunable — unlike the
# baseline script where it was fixed at 100.
PARAM_GRID = {
    "conv_filters":    [32, 64, 128],
    "kernel_size":     [2, 3, 5],
    "lstm_units":      [32, 64, 128],
    "dropout_rate":    [0.1, 0.2, 0.3],
    "learning_rate":   [0.0001, 0.001, 0.01],
    "batch_size":      [16, 32, 64],
    "lookback_window": [100],
}








# Data loading & feature engineering 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function creates the feature columns used by the CNN-LSTM.
# Categorical variables are converted into numeric codes where needed.
# Lagged log returns are added so the model can use recent return history.
# A rolling volatility feature is calculated using previous returns only.
#
# Other numeric columns are shifted by one period to avoid look-ahead bias.
# The horizon argument is used to exclude the correct target column.
# Rows with NaN values caused by shifting are removed.
# The returned dataframe is ready for sequence construction.
def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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

    Parameters
    df      : raw DataFrame (must already contain 'log_return' and the target col)
    horizon : forecast horizon (used only to correctly exclude the target column)
    """
    df = df.copy()

    # Step 1: Label-encode any non-numeric columns (except date and asset columns)
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

    # Lag all other numeric columns by 1 period to avoid look-ahead bias
    skip_cols = (
        {"log_return", f"target_log_return_t_plus_{horizon}"}
        | {f"lag_{l}" for l in LAG_FEATURES}
        | {"rolling_vol_5"}
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    extra_cols   = [c for c in numeric_cols if c not in skip_cols]

    for col in extra_cols:
        new_cols[f"{col}_lag1"] = df[col].shift(1)

    # Step 3: Concatenate all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Step 4: Drop rows with any NaN values introduced by shifting
    df = df.dropna().reset_index(drop=True)

    return df









# Load Bitcoin data and create the t+30 prediction target      
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function reads the raw Bitcoin CSV file and sorts it chronologically.
# The closing price column is validated, converted to numeric format,
# and invalid rows are removed.
#
# Daily log returns are calculated from the closing price.
# The t+30 target is created with log_return.shift(-30).
# This means the model predicts the single daily return 30 days ahead.
#
# Feature engineering is then applied and the cleaned dataframe is returned.
def load_data(csv_path: str, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Load the CSV file, compute log returns and the t+H target,
    then apply feature engineering.

    The horizon parameter controls:
      - The name of the target column: target_log_return_t_plus_{H}
      - Which column is excluded from the feature set in build_features()

    Parameters
    csv_path : path to the input CSV file
    horizon  : number of periods ahead to forecast (default: FORECAST_HORIZON)
    """
    logging.info("Reading %s  (horizon=t+%d)", csv_path, horizon)
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
    # shift(-H) moves future values into the current row — this is the
    # correct construction for a multi-step ahead forecasting target and
    # ensures compatibility with any horizon value.
    target_col = f"target_log_return_t_plus_{horizon}"
    df[target_col] = df["log_return"].shift(-horizon)

    # Drop rows where either the return or the target is NaN
    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    # Apply feature engineering (lags, rolling vol, lagged numeric columns)
    df = build_features(df, horizon)

    logging.info(
        "Loaded %d rows with %d features after engineering  (horizon=t+%d)",
        len(df), len(get_feature_cols(df, horizon)), horizon,
    )
    return df











# Select the columns used as CNN-LSTM input features      
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function returns the ordered list of model input columns.
# It excludes the date, close price, raw log return, asset column,
# and the target column for the selected forecast horizon.
#
# Any column starting with "target_" is also excluded as protection
# against accidental target leakage.
# The resulting feature list is used for scaling, sequence construction,
# training, validation, testing, and saved datasets.
def get_feature_cols(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> list:
    """
    Return the ordered list of feature columns used as model inputs.
    Excludes the target column, the date column, the raw price column,
    the raw log return column, and the 'asset' column if present.

    Parameters
    df      : feature-engineered DataFrame
    horizon : forecast horizon (used to correctly identify and exclude the target)
    """
    exclude = {
        DATE_COLUMN,
        PRICE_COLUMN,
        "log_return",
        f"target_log_return_t_plus_{horizon}",
        "asset",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith("target_")
    ]
    return feature_cols












# Train / validation / test split
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function divides the processed dataframe into chronological subsets.
# The split follows the predefined 70/15/15 ratio.
# The training set is used for model fitting and random search training.
# The validation set is used for leakage-free hyperparameter selection.
# The test set is reserved for final out-of-sample evaluation.
#
# No shuffling is applied because this is time-series forecasting.
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
        "Train: %d  |  Valid: %d  |  Test: %d",
        len(train), len(valid), len(test),
    )
    return train, valid, test









# Dataset persistence (unscaled, pre-sequence) 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function exports the processed splits to horizon-tagged CSV files.
# Only the date column, selected feature columns, and target column are saved.
# The data is saved before scaling, so future analysis can reuse raw features.
#
# Sequences are not saved, which keeps the datasets independent of lookback size.
# This supports later SHAP or permutation importance analysis.
# File names include the horizon tag so different runs stay separated.
def save_split_datasets(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list,
    target_col:   str,
    horizon:      int,
    output_dir:   str = OUTPUT_DIR,
) -> None:
    """
    Save the unscaled, post-feature-engineering tabular datasets to CSV.

    WHAT IS SAVED
    Each CSV contains:
      - All feature columns (identical to those passed to the model)
      - The target column
      - The date column (if present in the split DataFrame)
      - No scaling is applied — raw feature values are written to disk

    This design means:
      • The saved files exactly reproduce the arrays consumed by the model
        (before the StandardScaler transform).
      • Feature importance methods (SHAP, permutation) can re-apply the
        same scaler (loaded from disk) to reproduce the model's exact inputs.
      • Different lookback windows can still be tested without regenerating
        these files, because sequences are NOT included.
      • No data leakage: the split was performed chronologically before saving.

    FILE NAMING
    Files are named train_t{H}.csv, valid_t{H}.csv, test_t{H}.csv where H
    is the forecast horizon, so outputs from different horizons do not
    overwrite each other.

    Parameters
    train_df     : training split DataFrame (post-feature-engineering)
    valid_df     : validation split DataFrame
    test_df      : test split DataFrame
    feature_cols : ordered list of feature column names
    target_col   : name of the target column
    horizon      : forecast horizon (used in the output filename)
    output_dir   : directory where files are written
    """
    os.makedirs(output_dir, exist_ok=True)

    # Columns to keep: date (if present) + features + target
    date_cols = [DATE_COLUMN] if DATE_COLUMN in train_df.columns else []
    keep_cols = date_cols + feature_cols + [target_col]

    splits = {
        f"train_t{horizon}": train_df,
        f"valid_t{horizon}": valid_df,
        f"test_t{horizon}":  test_df,
    }

    for name, split_df in splits.items():
        # Keep only the columns that exist in this split DataFrame
        cols_present = [c for c in keep_cols if c in split_df.columns]
        out_path = os.path.join(output_dir, f"{name}.csv")
        split_df[cols_present].reset_index(drop=True).to_csv(out_path, index=False)
        logging.info(
            "Saved dataset  %-20s  →  %s  (%d rows × %d cols)",
            name, out_path, len(split_df), len(cols_present),
        )








# Model and scaler persistence 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function saves the final CNN-LSTM model and StandardScaler to disk.
# The model is saved as a Keras .keras file.
# The scaler is saved with joblib.
#
# Both artefacts correspond to the final train+validation setup.
# Saving them allows later SHAP or permutation importance analysis without
# retraining the model or refitting the scaler.
# The horizon is included in filenames for traceability.
def save_model_and_scaler(
    model:      Sequential,
    scaler:     StandardScaler,
    horizon:    int,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Persist the trained CNN–LSTM model and the fitted scaler to disk.

    Feature importance methods (SHAP, permutation importance) require the
    trained model and the exact scaler used during training.  Saving both
    artefacts means they can be loaded later without retraining, and the
    full pipeline (load CSV → apply scaler → build sequences → predict)
    can be reproduced exactly.

    OUTPUT FILES
    cnnlstm_model_t{H}.keras  — the Keras model in the native .keras format
                                  (recommended over .h5 for Keras 3.x)
    cnnlstm_scaler_t{H}.joblib — the fitted StandardScaler, serialised with
                                   joblib (efficient for NumPy-backed objects)

    Parameters
    model      : final trained Keras Sequential model
    scaler     : StandardScaler fitted on train + validation data
    horizon    : forecast horizon (used in filename)
    output_dir : directory where artefacts are written
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path  = os.path.join(output_dir, f"cnnlstm_model_t{horizon}.keras")
    scaler_path = os.path.join(output_dir, f"cnnlstm_scaler_t{horizon}.joblib")

    model.save(model_path)
    logging.info("Saved model  → %s", model_path)

    joblib.dump(scaler, scaler_path)
    logging.info("Saved scaler → %s", scaler_path)








# Metrics (identical to baseline) 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

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






# Calculate directional accuracy                
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function measures whether the model predicts the correct return direction.
# It compares the sign of the predicted log return with the sign of the actual
# future log return.
#
# A prediction is counted as correct when both signs match.
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of predictions where the model correctly predicted
    whether the return would be positive or negative.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0)






# Calculate model evaluation metrics        
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function computes the main evaluation metrics for the CNN-LSTM forecasts.
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








# Sequence construction 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function converts a 2-D scaled feature matrix into 3-D sequences.
# Each sequence contains the previous `lookback` observations.
# The lookback window is passed as an argument because it is tunable here.
#
# The target for each sequence is the value directly after the input window.
# Only historical rows enter each sequence, preventing future leakage.
# The saved CSVs remain tabular; sequence creation happens only inside modelling.
# If too few rows exist, empty arrays are returned.
def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """
    Convert a 2-D scaled feature array into overlapping 3-D sequences
    using the given lookback window size.

    NOTE ON DESIGN
    ──────────────
    Sequence construction is intentionally kept INSIDE the modelling pipeline
    and is NOT applied to the saved CSV datasets.  This means:

      • The saved CSVs are lookback-agnostic: you can test a lookback of 50,
        100, or 200 without regenerating the feature files.
      • Feature importance analysis on the test CSV does not require knowledge
        of the lookback window used during training.
      • The tabular format is directly compatible with tree-based importance
        methods (e.g. Random Forest, XGBoost) that do not use sequences.

    For each index i >= lookback the input sequence spans rows
    [i - lookback : i], and the target is y[i].  Only past information
    is used; no future rows enter any sequence.

    Parameters
    X        : shape (T, F) — scaled feature array
    y        : shape (T,)   — target array
    lookback : number of time steps to include in each input window

    Returns
    X_seq : shape (N, lookback, F)
    y_seq : shape (N,)
    """
    n = len(X)

    # Not enough rows to form even one sequence
    if n <= lookback:
        empty_X = np.empty((0, lookback, X.shape[1]), dtype=np.float32)
        empty_y = np.empty(0, dtype=np.float32)
        return empty_X, empty_y

    # Number of valid sequences
    N = n - lookback

    # Pre-allocate arrays for efficiency
    X_seq = np.empty((N, lookback, X.shape[1]), dtype=np.float32)
    y_seq = np.empty(N, dtype=np.float32)

    # Build each sequence one by one (explicit loop for clarity)
    for k in range(N):
        X_seq[k] = X[k : k + lookback]
        y_seq[k] = y[k + lookback]

    return X_seq, y_seq







# CNN–LSTM model builder 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function defines the hybrid CNN-LSTM model architecture.
# Conv1D first extracts local temporal patterns from the lookback window.
# Causal padding prevents future information from entering the convolution.
# MaxPooling1D reduces the temporal dimension before the LSTM layer.
#
# The LSTM layer captures longer sequential dependencies.
# Dropout is used for regularisation.
# A single dense linear output is used for regression.
# The model is compiled with Adam and MSE loss.
def build_cnnlstm(
    n_features:    int,
    lookback:      int,
    filters:       int,
    kernel_size:   int,
    lstm_units:    int,
    dropout_rate:  float,
    learning_rate: float,
) -> Sequential:
    """
    Build and compile a CNN–LSTM Sequential model.

    Architecture:
      Input(lookback, n_features)
      → Conv1D(filters, kernel_size, activation='relu', padding='causal')
      → MaxPooling1D(pool_size=2)
      → LSTM(lstm_units)
      → Dropout(dropout_rate)
      → Dense(1, linear activation)

    Design notes:
      - 'causal' padding on Conv1D ensures no future time steps leak into
        the convolution output, preserving the no-look-ahead guarantee.
      - pool_size is fixed at 2 (not tuned) to keep the search space manageable.
      - The linear output activation is appropriate for regression.
      - MSE loss is consistent with a regression objective.

    Parameters
    n_features    : number of input features per time step
    lookback      : number of time steps in each input sequence
    filters       : number of convolutional filters
    kernel_size   : width of the 1-D convolutional kernel (time steps)
    lstm_units    : number of LSTM memory units
    dropout_rate  : fraction of units dropped for regularisation
    learning_rate : step size for the Adam optimiser

    Returns
    Compiled Keras Sequential model.
    """
    model = Sequential()

    # Explicit Input layer (recommended by Keras 3.x)
    model.add(Input(shape=(lookback, n_features)))

    # Conv1D: extracts local temporal patterns from sub-windows.
    # 'causal' padding prevents any right-side (future) context from leaking in.
    model.add(Conv1D(
        filters     = filters,
        kernel_size = kernel_size,
        activation  = "relu",
        padding     = "causal",
    ))

    # MaxPooling1D: reduces the temporal dimension before passing to the LSTM.
    # pool_size is fixed at 2 — it is not part of the random search.
    model.add(MaxPooling1D(pool_size=2))

    # LSTM: captures long-range sequential dependencies in the pooled features
    model.add(LSTM(lstm_units))

    # Dropout for regularisation (reduces overfitting)
    model.add(Dropout(dropout_rate))

    # Single linear output unit (regression)
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "mse",
    )
    return model












# Initial training (with early stopping) 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function builds and trains a CNN-LSTM model using one parameter set.
# It trains on pre-built sequences created with the candidate lookback window.
# If enough sequences are available, the final EARLY_STOP_WINDOW sequences
# are used as a trailing validation window for early stopping.
#
# Early stopping stops training when validation loss no longer improves.
# This reduces overfitting during the initial training phase.
# If too few sequences exist, the model trains on all available sequences.
# The trained model is returned for validation or online evaluation.
def initial_train(
    X_seq:  np.ndarray,
    y_seq:  np.ndarray,
    params: dict,
) -> Sequential:
    """
    Build and perform the initial full training of the CNN–LSTM on a sequence
    dataset, with early stopping on a held-out trailing validation window.

    The trailing EARLY_STOP_WINDOW sequences are held out as the early-stop
    validation set.  This prevents overfitting during the offline training phase
    before online updates begin.

    Parameters
    X_seq  : (N, lookback, F) sequences
    y_seq  : (N,) corresponding targets
    params : hyperparameter dictionary — must contain conv_filters, kernel_size,
             lstm_units, dropout_rate, learning_rate, batch_size, lookback_window

    Returns
    Trained Keras Sequential model.
    """
    n_features = X_seq.shape[2]
    lookback   = X_seq.shape[1]

    # Build the model using the provided hyperparameters
    model = build_cnnlstm(
        n_features    = n_features,
        lookback      = lookback,
        filters       = params["conv_filters"],
        kernel_size   = params["kernel_size"],
        lstm_units    = params["lstm_units"],
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











# Online evaluation loop 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function simulates streaming model evaluation.
# For each new evaluation row, the model predicts using the latest
# `lookback` rows from the rolling context buffer.
# After prediction, the realised target is used to update the model
# through train_on_batch.
#
# The new feature row is then added to the context buffer.
# This allows the model to adapt as new observations arrive.
# The lookback is explicit because it is tuned during random search.
def online_evaluate(
    model:            Sequential,
    X_eval_scaled:    np.ndarray,
    y_eval:           np.ndarray,
    batch_size:       int,
    X_context_scaled: np.ndarray,
    lookback:         int,
):
    """
    Online predict-then-update loop over an evaluation block.

    For each observation i in the evaluation set:
      1. Construct the prediction sequence from the last lookback rows of
         the combined context (training history + evaluation rows seen so far).
      2. Predict t+H log return.
      3. Update model weights via train_on_batch on the new (sequence, target).
      4. Absorb the new row into the rolling context buffer.

    Note: lookback is passed in as an explicit argument here because it is
    now a tunable hyperparameter and may differ between candidates.

    Parameters
    model             : pre-trained Keras CNN–LSTM model
    X_eval_scaled     : (M, F) scaled features for the evaluation block
    y_eval            : (M,) targets for the evaluation block
    batch_size        : mini-batch size for train_on_batch
    X_context_scaled  : (T, F) scaled features from the history before this block
    lookback          : number of time steps used to build each prediction window

    Returns
    predictions : numpy array of predicted log returns
    actuals     : numpy array of actual log returns
    """
    predictions = []
    actuals     = []

    # The context buffer starts with the training history.
    # It grows by one row after each step as new observations are absorbed.
    context = X_context_scaled.copy()

    for i in range(len(X_eval_scaled)):

        # If we do not have enough history yet, just absorb the row and move on
        if len(context) < lookback:
            new_row = X_eval_scaled[[i]]
            context = np.vstack([context, new_row])
            continue

        # Step 1: Build the prediction sequence from the last lookback rows
        x_seq = context[-lookback:][np.newaxis, :, :].astype(np.float32)

        # Step 2: Predict the t+H log return
        pred = float(model.predict(x_seq, verbose=0)[0, 0])
        predictions.append(pred)
        actuals.append(float(y_eval[i]))

        # Step 3: Online update — adjust model weights using the newly revealed actual
        y_new = np.array([[y_eval[i]]], dtype=np.float32)
        model.train_on_batch(x_seq, y_new)

        # Step 4: Absorb the new row into the context buffer
        new_row = X_eval_scaled[[i]]
        context = np.vstack([context, new_row])

    predictions_array = np.asarray(predictions, dtype=float)
    actuals_array     = np.asarray(actuals,     dtype=float)

    return predictions_array, actuals_array










# Leakage-free validation loop (V3) 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function is similar to online_evaluate, but does not call train_on_batch.
# It is used during random search validation to avoid data leakage.
#
# The model predicts sequentially using the rolling context buffer.
# New validation feature rows are added to the context after each prediction.
# However, validation targets never update the model weights.
# This makes hyperparameter selection based on genuinely unseen validation data.
def predict_only_evaluate(
    model:            Sequential,
    X_eval_scaled:    np.ndarray,
    y_eval:           np.ndarray,
    X_context_scaled: np.ndarray,
    lookback:         int,
):
    """
    Online-style prediction loop over an evaluation block with NO weight updates.

    Used exclusively during random search (hyperparameter optimisation) to
    evaluate each candidate model on the validation set without leaking
    validation targets into the model weights.

    This function is structurally identical to online_evaluate with one
    deliberate omission: the train_on_batch call is removed.  Validation
    targets are therefore never used to update model weights during the search,
    eliminating the data leakage that was present in V2.

    The rolling context mechanism is preserved exactly:
      • The context buffer is seeded with the scaled training history.
      • After each prediction the new feature row is absorbed into the buffer
        so that the next prediction sees an up-to-date rolling window.
      • Only the last `lookback` rows of the buffer are used per prediction.

    Contrast with online_evaluate (used on the training and test splits inside
    fit_and_evaluate), which performs a train_on_batch step after each
    prediction to simulate true online learning.

    Parameters
    model             : pre-trained Keras CNN–LSTM model
    X_eval_scaled     : (M, F) scaled features for the evaluation block
    y_eval            : (M,) targets for the evaluation block
    X_context_scaled  : (T, F) scaled features from the history before this block
    lookback          : number of time steps used to build each prediction window

    Returns
    predictions : numpy array of predicted log returns
    actuals     : numpy array of actual log returns
    """
    predictions = []
    actuals     = []

    # The context buffer starts with the training history.
    # It grows by one row after each step as new feature rows are absorbed.
    # Only the last `lookback` rows are ever needed for a prediction.
    context = X_context_scaled.copy()

    for i in range(len(X_eval_scaled)):

        # If we do not have enough history yet, just absorb the row and move on
        if len(context) < lookback:
            new_row = X_eval_scaled[[i]]
            context = np.vstack([context, new_row])
            continue

        # Step 1: Build the prediction sequence from the last lookback rows
        x_seq = context[-lookback:][np.newaxis, :, :].astype(np.float32)

        # Step 2: Predict the t+H log return
        pred = float(model.predict(x_seq, verbose=0)[0, 0])
        predictions.append(pred)
        actuals.append(float(y_eval[i]))

        # Step 3: Absorb the new feature row into the context buffer.
        # NOTE: train_on_batch is deliberately NOT called here.  Validation
        # targets must not influence model weights during hyperparameter search.
        new_row = X_eval_scaled[[i]]
        context = np.vstack([context, new_row])

    predictions_array = np.asarray(predictions, dtype=float)
    actuals_array     = np.asarray(actuals,     dtype=float)

    return predictions_array, actuals_array











# Scaler helper 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function fits a StandardScaler on the provided feature matrix.
# It returns both the fitted scaler and the scaled data.
#
# The scaler must only be fitted on training data or train+validation data.
# Validation and test data should only be transformed with an already fitted scaler.
# This avoids distributional leakage from future periods.
# The fitted scaler is also saved later for reproducible feature importance work.
def fit_scaler(X: np.ndarray):
    """
    Fit a StandardScaler on X and return the fitted scaler and the scaled array.

    CRITICAL SCALING RULE
    The scaler must ALWAYS be fit on training data only.  Fitting on validation
    or test data would introduce data leakage — the model would implicitly
    receive information about future distributions during training.

    The fitted scaler is later used to transform validation and test data
    (transform only, no re-fitting), ensuring a consistent normalisation
    that reflects only the statistics of the training period.

    This function is a thin wrapper that makes the fit-transform pair
    explicit and makes the scaler object available for saving to disk.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    return scaler, X_scaled









# Random configuration sampler 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function draws one random value from each entry in PARAM_GRID.
# The result is a single CNN-LSTM configuration.
# This includes architecture settings, training settings, and lookback_window.
#
# Random sampling is used instead of exhaustive grid search to reduce runtime.
# The returned dictionary can directly be passed into training and evaluation.
# This function is repeatedly called inside random_search().
def sample_params() -> dict:
    """
    Draw one random configuration by sampling one value from each list
    in PARAM_GRID independently and uniformly at random.

    Returns a plain dictionary with one value per hyperparameter.
    """
    config = {}
    for param_name, choices in PARAM_GRID.items():
        config[param_name] = random.choice(choices)
    return config








# Check whether a sampled CNN-LSTM configuration is valid        
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function checks whether the sampled configuration can safely be used.
# The main concern is whether the lookback window is large enough for the
# Conv1D and MaxPooling1D operations.
#
# It rejects configurations where lookback is too small.
# It also rejects cases where the kernel size is larger than the lookback.
# This prevents avoidable Keras shape errors during model construction.
# Valid configurations return True, invalid ones return False.
def is_valid_config(params: dict, lookback: int) -> bool:
    """
    Check whether a configuration is geometrically valid for the CNN layer.

    With causal padding the Conv1D output length equals the input length,
    so the kernel_size itself is always safe.  However, MaxPooling1D with
    pool_size=2 requires the post-conv sequence to have at least 2 steps.
    A lookback of at least 2 is therefore the only hard constraint.

    If any additional incompatibility is detected (e.g. the kernel is larger
    than the lookback), the config is marked invalid and skipped.

    Returns True if the config is safe to use, False if it should be skipped.
    """
    # The input to MaxPooling1D has length == lookback (causal padding preserves length).
    # We need at least pool_size steps after conv to feed the LSTM.
    if lookback < 2:
        return False

    # kernel_size must not exceed the lookback window
    if params["kernel_size"] > lookback:
        return False

    return True











# Random search 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function evaluates randomly sampled CNN-LSTM configurations.
# For each draw, it samples parameters, checks whether the setup is valid,
# builds sequences using the sampled lookback, and trains the model.
#
# Validation is performed with predict_only_evaluate, so validation targets
# are not used to update model weights.
# Each candidate is scored using validation RMSE and MAE.
# The configuration with the lowest validation RMSE is selected as best.
# All evaluated configurations are returned for saving.
def random_search(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    feature_cols: list,
    target_col:   str,
) -> tuple:
    """
    Run a random search over N_RANDOM_DRAWS configurations.

    Each candidate configuration is:
      1. Trained on the training split (with early stopping).
      2. Evaluated on the validation split using predict_only_evaluate —
         predictions are generated sequentially from the rolling context, but
         NO weight updates (train_on_batch) occur.  Validation targets therefore
         never influence model weights, eliminating the leakage present in V2.
      3. Scored by validation RMSE.

    The configuration with the lowest validation RMSE is returned as the best.

    The scaler is fit on the training data once and reused across all candidates.
    This is correct because the training data does not change between candidates
    — only the model architecture and hyperparameters change.

    Parameters
    train_df     : training split DataFrame
    valid_df     : validation split DataFrame
    feature_cols : list of feature column names
    target_col   : name of the target column

    Returns
    best_params  : dict of the best hyperparameter configuration found
    all_results  : list of dicts, one per evaluated candidate, with params + metrics
    """
    all_results = []   # Will collect one row per evaluated candidate

    best_rmse   = float("inf")
    best_params = None

    logging.info("Starting random search — %d iterations", N_RANDOM_DRAWS)

    # Pull raw arrays from the training split once (same for all candidates)
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train     = train_df[target_col].values.astype(np.float32)

    # Pull raw arrays from the validation split once
    X_valid_raw = valid_df[feature_cols].values.astype(np.float32)
    y_valid     = valid_df[target_col].values.astype(np.float32)

    # Fit the scaler on training data only — used for all candidates
    scaler_search, X_train_scaled = fit_scaler(X_train_raw)

    # Scale validation data using the same scaler (no re-fitting)
    X_valid_scaled = scaler_search.transform(X_valid_raw)

    draw_number = 0

    while draw_number < N_RANDOM_DRAWS:

        # Sample a random configuration
        params = sample_params()
        lookback = params["lookback_window"]

        # Skip this config if it would cause a Keras error
        if not is_valid_config(params, lookback):
            logging.info(
                "  Draw skipped — invalid config (kernel_size=%d > lookback=%d)",
                params["kernel_size"], lookback,
            )
            continue

        draw_number += 1
        logging.info(
            "  Draw %d / %d : %s",
            draw_number, N_RANDOM_DRAWS, params,
        )

        # Build sequences using this candidate's lookback window
        X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train, lookback)

        # If there are not enough sequences to train on, skip this config
        if len(X_train_seq) == 0:
            logging.info("  Draw %d skipped — not enough training sequences", draw_number)
            row = dict(params)
            row["val_rmse"] = float("nan")
            row["val_mae"]  = float("nan")
            all_results.append(row)
            continue

        # Train the candidate model on the training sequences
        try:
            model = initial_train(X_train_seq, y_train_seq, params)
        except Exception as exc:
            logging.warning("  Draw %d failed during training: %s", draw_number, exc)
            row = dict(params)
            row["val_rmse"] = float("nan")
            row["val_mae"]  = float("nan")
            all_results.append(row)
            continue

        # V3: evaluate the candidate on the validation split using predict_only_evaluate.
        # This generates predictions sequentially via the rolling context but does NOT
        # call train_on_batch, so validation targets never update model weights.
        val_preds, val_actuals = predict_only_evaluate(
            model            = model,
            X_eval_scaled    = X_valid_scaled,
            y_eval           = y_valid,
            X_context_scaled = X_train_scaled,
            lookback         = lookback,
        )

        # If the online loop skipped all rows (not enough context), record NaN
        if len(val_preds) == 0:
            logging.info("  Draw %d — no valid predictions on validation set", draw_number)
            row = dict(params)
            row["val_rmse"] = float("nan")
            row["val_mae"]  = float("nan")
            all_results.append(row)
            continue

        # Compute validation metrics for this candidate
        val_rmse = sqrt(mean_squared_error(val_actuals, val_preds))
        val_mae  = mean_absolute_error(val_actuals, val_preds)

        logging.info(
            "  Draw %d — val RMSE: %.8f  |  val MAE: %.8f",
            draw_number, val_rmse, val_mae,
        )

        # Store the result for this candidate
        row = dict(params)
        row["val_rmse"] = val_rmse
        row["val_mae"]  = val_mae
        all_results.append(row)

        # Update the best configuration if this one is better
        if val_rmse < best_rmse:
            best_rmse   = val_rmse
            best_params = dict(params)
            logging.info(
                "  --> New best found at draw %d  (val RMSE: %.8f)",
                draw_number, best_rmse,
            )

    logging.info(
        "Random search complete.  Best val RMSE: %.8f  |  Best params: %s",
        best_rmse, best_params,
    )

    return best_params, all_results








# Final model evaluation (train split and test split)
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function evaluates the selected best parameter configuration.
# The tuned lookback window is taken from the best parameter dictionary.
# For the training split, a warm-up section is used before online evaluation.
#
# For the test split, the model is retrained on train+validation data.
# The scaler is fitted only on train+validation features.
# The test set is then evaluated using online predict-then-update learning.
# The final trained model and scaler are saved and returned.
def fit_and_evaluate(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    params:       dict,
    feature_cols: list,
    target_col:   str,
    horizon:      int,
):
    """
    Evaluate the best hyperparameter configuration on both the training
    split (in-sample) and the held-out test set (out-of-sample).

    This function mirrors the same structure as the baseline script but uses
    the best_params from the random search rather than fixed hyperparameters.
    lookback_window is read from params instead of from a global constant.

    After final training on train + validation:
      - The model is saved as cnnlstm_model_t{H}.keras
      - The scaler (fit on train+valid) is saved as cnnlstm_scaler_t{H}.joblib

    IN-SAMPLE EVALUATION (training split)
    The model is trained on the first (lookback + EARLY_STOP_WINDOW + 1)
    rows as the warm-up, then the online loop runs over the remainder of
    the training split.

    OUT-OF-SAMPLE EVALUATION (test split)
    The scaler is re-fit on train + validation features.  The model is
    retrained from scratch on all sequences from train + validation data.
    The online loop then runs over the test split: predict -> train_on_batch.

    Parameters
    train_df     : training split DataFrame
    valid_df     : validation split DataFrame
    test_df      : test split DataFrame
    params       : best hyperparameter configuration from random search
    feature_cols : ordered list of feature column names
    target_col   : name of the target column
    horizon      : forecast horizon (used when saving model/scaler filenames)

    Returns
    (test_pred, test_actual, test_metrics, train_pred, train_actual, train_metrics,
     final_model, scaler_tv)
    The final model and scaler are returned so the caller can optionally use them
    (e.g. for further persistence or analysis) in addition to being saved to disk.
    """
    lookback = params["lookback_window"]
    warmup   = lookback + EARLY_STOP_WINDOW + 1

    # In-sample: online loop over the training split
    train_init = train_df.iloc[:warmup]
    train_eval = train_df.iloc[warmup:]

    # Fit the scaler on the warm-up portion only (no leakage)
    X_init_raw           = train_init[feature_cols].values.astype(np.float32)
    y_init               = train_init[target_col].values.astype(np.float32)
    scaler_train, X_init_scaled = fit_scaler(X_init_raw)

    # Scale the rest of the training set using the same scaler (no re-fitting)
    X_train_eval_raw    = train_eval[feature_cols].values.astype(np.float32)
    X_train_eval_scaled = scaler_train.transform(X_train_eval_raw)
    y_train_eval        = train_eval[target_col].values.astype(np.float32)

    # Build sequences from the warm-up data and do the initial offline training
    X_init_seq, y_init_seq = build_sequences(X_init_scaled, y_init, lookback)
    train_model = initial_train(X_init_seq, y_init_seq, params)

    # Run the online loop over the remaining training rows
    train_pred, train_actual = online_evaluate(
        model            = train_model,
        X_eval_scaled    = X_train_eval_scaled,
        y_eval           = y_train_eval,
        batch_size       = params["batch_size"],
        X_context_scaled = X_init_scaled,
        lookback         = lookback,
    )

    # Compute in-sample metrics
    train_metrics = calculate_metrics(train_actual, train_pred)

    # Out-of-sample: retrain on train + validation, then online over test 
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Re-fit the scaler on train + validation (no leakage into the test set)
    X_tv_raw             = train_valid_df[feature_cols].values.astype(np.float32)
    y_tv                 = train_valid_df[target_col].values.astype(np.float32)
    scaler_tv, X_tv_scaled = fit_scaler(X_tv_raw)

    # Build sequences and train the final model on train + validation data
    X_tv_seq, y_tv_seq = build_sequences(X_tv_scaled, y_tv, lookback)
    test_model = initial_train(X_tv_seq, y_tv_seq, params)

    # Scale the test features using the train+validation scaler (no re-fitting)
    X_test_raw    = test_df[feature_cols].values.astype(np.float32)
    X_test_scaled = scaler_tv.transform(X_test_raw)
    y_test        = test_df[target_col].values.astype(np.float32)

    # Run the online loop over the test set
    test_pred, test_actual = online_evaluate(
        model            = test_model,
        X_eval_scaled    = X_test_scaled,
        y_eval           = y_test,
        batch_size       = params["batch_size"],
        X_context_scaled = X_tv_scaled,
        lookback         = lookback,
    )

    # Compute out-of-sample metrics
    test_metrics = calculate_metrics(test_actual, test_pred)

    # Persist the final model and scaler
    # test_model was trained on train+validation (the correct final model).
    # scaler_tv was fit on the same data — it is the scaler to save.
    save_model_and_scaler(
        model      = test_model,
        scaler     = scaler_tv,
        horizon    = horizon,
        output_dir = OUTPUT_DIR,
    )

    return (test_pred, test_actual, test_metrics,
            train_pred, train_actual, train_metrics,
            test_model, scaler_tv)










# Output
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function exports all evaluated random search configurations.
# Each row contains one parameter setup and its validation RMSE and MAE.
# The results are sorted by validation RMSE so the best candidates appear first.
#
# This file provides a record of the optimisation process.
# It can be used later to inspect which hyperparameters performed best.
# The output filename reflects the CNN-LSTM after-hyper experiment.
def save_search_results(all_results: list) -> None:
    """
    Save all random search candidate results to a CSV file.

    Each row represents one evaluated configuration, including its
    hyperparameters, validation RMSE, and validation MAE.
    """
    results_df = pd.DataFrame(all_results)

    # Put the metrics columns first for easy reading
    cols = ["val_rmse", "val_mae"] + [c for c in results_df.columns if c not in ("val_rmse", "val_mae")]
    results_df = results_df[cols]

    # Sort by validation RMSE so the best candidates appear at the top
    results_df = results_df.sort_values("val_rmse").reset_index(drop=True)

    output_filename = "cnnlstm_random_search_results_after_hyper_1.csv"
    results_df.to_csv(output_filename, index=False)
    logging.info("Saved random search results → %s", output_filename)










# Save final test predictions to CSV                
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function exports the final out-of-sample test predictions.
# The predictions are aligned with the original test rows from the full dataframe.
# Actual and predicted log returns are stored using horizon-aware column names.
#
# The best hyperparameters are added to the prediction file for traceability.
# The output filename reflects the optimisation status and forecast horizon.
# This CSV can later be used for plotting, reporting, and model comparison.
def save_test_predictions(
    full_df:     pd.DataFrame,
    train_len:   int,
    valid_len:   int,
    test_pred:   np.ndarray,
    test_actual: np.ndarray,
    params:      dict,
    target_col:  str,
    horizon:     int,
) -> None:
    """
    Save the final test predictions to a CSV file.

    The output format mirrors the baseline script so that results can be
    compared directly.  The filename reflects the "after" optimisation status
    and the forecast horizon.  The best hyperparameters found by the random
    search are also recorded in each row for full traceability.

    Parameters
    full_df    : complete feature-engineered DataFrame (all splits combined)
    train_len  : number of rows in the training split
    valid_len  : number of rows in the validation split
    test_pred  : array of predicted log returns for the test split
    test_actual: array of actual log returns for the test split
    params     : best hyperparameter configuration
    target_col : name of the target column
    horizon    : forecast horizon (used in the output filename)
    """
    # The test predictions start immediately after the train + validation rows
    start_idx = train_len + valid_len
    pred_df   = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()

    # Add prediction columns (use horizon-aware column names)
    pred_df[f"actual_log_return_t_plus_{horizon}"]    = test_actual
    pred_df[f"predicted_log_return_t_plus_{horizon}"] = test_pred

    # Record the best hyperparameters for traceability
    for key, value in params.items():
        pred_df[f"best_{key}"] = value

    output_filename = f"cnnlstm_test_predictions_after_hyper_t{horizon}.csv"
    pred_df.to_csv(output_filename, index=False)
    logging.info("Saved test predictions → %s", output_filename)










# Main 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This is the main controller function for the experiment.
# It loads data, creates the t+30 target, builds features, and splits the data.
# The unscaled split datasets are saved for reproducibility and SHAP analysis.
#
# Random search is performed to find the best CNN-LSTM hyperparameters.
# The best model is retrained and evaluated on train and test data.
# The model, scaler, random search results, and test predictions are saved.
# Results are also logged and printed in a clear terminal format.
def cnnlstm_forecast(horizon: int = FORECAST_HORIZON) -> None:
    """
    Main entry point.  Orchestrates the full pipeline:
      1.  Load and preprocess data for the given forecast horizon
      2.  Split into train / validation / test
      3.  Save unscaled split datasets to CSV (for reproducibility & feature importance)
      4.  Run the random search on the train + validation splits
      5.  Save the search results
      6.  Retrain and evaluate the best model on both train and test splits
      7.  Save the final model and scaler to disk
      8.  Print results to the terminal
      9.  Save test predictions to CSV

    HORIZON PARAMETERISATION
    The `horizon` argument makes the entire pipeline horizon-aware:
      - load_data() constructs shift(-horizon) targets
      - build_features() excludes the correct target column
      - get_feature_cols() excludes the correct target column
      - save_split_datasets() names output files train_t{H}.csv etc.
      - save_model_and_scaler() names artefacts cnnlstm_model_t{H}.keras etc.
      - save_test_predictions() names the predictions CSV accordingly


    Parameters
    horizon : number of periods ahead to forecast (default: FORECAST_HORIZON)
    """

    logging.info("=" * 70)
    logging.info("CNN–LSTM forecast pipeline  |  horizon = t+%d", horizon)
    logging.info("=" * 70)

    # Step 1: Load data, compute log returns, build features
    # The full feature engineering pipeline (lags, rolling vol, numeric lags)
    # is executed once per horizon.  The target is always shift(-horizon).
    df           = load_data(CSV_PATH, horizon=horizon)
    target_col   = f"target_log_return_t_plus_{horizon}"
    feature_cols = get_feature_cols(df, horizon=horizon)

    logging.info(
        "Feature columns (%d total): %s", len(feature_cols), feature_cols
    )

    # Step 2: Chronological split 
    # Splitting is performed BEFORE saving so that each CSV represents a clean,
    # non-overlapping, temporally ordered partition of the data.
    train_df, valid_df, test_df = split_data(df)

    # Step 3: Save unscaled split datasets 
    # Datasets are saved HERE — after feature engineering, after splitting,
    # and BEFORE any scaling.  This is the correct point in the pipeline:
    #
    #   - All feature engineering has been applied (lags, rolling vol, encoding)
    #   - The split is chronological with no data leakage
    #   - No scaling has been applied — scaling must be reproduced from the
    #     saved scaler (cnnlstm_scaler_t{H}.joblib) when reusing these files
    #   - Sequences have NOT been constructed — lookback window is irrelevant
    #     at this stage and can be varied without regenerating these CSVs
    #
    # Files written: train_t{H}.csv, valid_t{H}.csv, test_t{H}.csv
    save_split_datasets(
        train_df     = train_df,
        valid_df     = valid_df,
        test_df      = test_df,
        feature_cols = feature_cols,
        target_col   = target_col,
        horizon      = horizon,
        output_dir   = OUTPUT_DIR,
    )

    # Step 4: Run the random search 
    logging.info(
        "Running random search with %d draws over param grid: %s",
        N_RANDOM_DRAWS, PARAM_GRID,
    )

    best_params, all_results = random_search(
        train_df     = train_df,
        valid_df     = valid_df,
        feature_cols = feature_cols,
        target_col   = target_col,
    )

    # Step 5: Save the full search results to CSV 
    save_search_results(all_results)

    # Step 6: Log the best configuration that was found 
    logging.info("Best params selected: %s", best_params)

    # Step 7: Retrain the best model and evaluate on train + test splits
    # fit_and_evaluate() also saves the model (.keras) and scaler (.joblib).
    (test_pred, test_actual, test_metrics,
     train_pred, train_actual, train_metrics,
     final_model, final_scaler) = fit_and_evaluate(
        train_df     = train_df,
        valid_df     = valid_df,
        test_df      = test_df,
        params       = best_params,
        feature_cols = feature_cols,
        target_col   = target_col,
        horizon      = horizon,
    )

    # Step 8: Log results to the terminal 
    logging.info(
        "----- TRAIN RESULTS (online learning, t+%d) -----", horizon
    )
    for metric_name, metric_value in train_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info(
        "----- FINAL TEST RESULTS (online learning, t+%d) -----", horizon
    )
    for metric_name, metric_value in test_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    # Step 9: Print a formatted results block to the console 
    print("\n" + "=" * 60)
    print(f"  TRAIN RESULTS  —  CNN–LSTM Online Learning (t+{horizon} forecast)")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Lookback window         : {best_params['lookback_window']} (tuned)")
    print(f"  Forecast horizon        : t+{horizon}")
    print(f"  MAE                     : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {train_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"  FINAL TEST RESULTS  —  CNN–LSTM Online Learning (t+{horizon} forecast)")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Lookback window         : {best_params['lookback_window']} (tuned)")
    print(f"  Forecast horizon        : t+{horizon}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    # Step 10: Save test predictions to CSV 
    save_test_predictions(
        full_df     = df,
        train_len   = len(train_df),
        valid_len   = len(valid_df),
        test_pred   = test_pred,
        test_actual = test_actual,
        params      = best_params,
        target_col  = target_col,
        horizon     = horizon,
    )

    logging.info("Pipeline complete for horizon=t+%d", horizon)


if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
    )

    # Single horizon (default) 
    # Run for the horizon defined by FORECAST_HORIZON at the top of the script.
    cnnlstm_forecast(horizon=FORECAST_HORIZON)
