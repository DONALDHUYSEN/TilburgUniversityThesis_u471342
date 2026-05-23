"""
lstm_btc_log_returns_before_hyper_1.py
────────────────────────────────────────
LSTM model for Bitcoin log-return forecasting at a t+1 horizon.
Online learning variant with a fixed 100-day lookback window.

THIS SCRIPT
───────────────────────
This is the "before hyperparameter optimisation" version.
It uses one single, manually chosen set of hyperparameters rather than
running any search procedure.  The goal is to establish a baseline that
can be compared directly with the optimised version (which selects
hyperparameters via random search on the validation set).

Everything else — data loading, feature engineering, the train/validation/
test split, the online learning loop, the metric set, and the output format
— is kept identical to the other scripts
so that results are methodologically comparable.

MODEL DESCRIPTION
─────────────────
A Long Short-Term Memory (LSTM) network is used for sequence-to-one
regression, forecasting the next-period (t+1) log return.

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
───────────────────────────────────
Phase 1 — Initial training (train split):
  The scaler is fit on the training data.  The model is trained on all
  sequences derived from the scaled training set, with early stopping
  monitored on a trailing validation window carved from the training
  sequences.

Phase 2 — Skipped in this version:
  No hyperparameter search is performed.  The fixed hyperparameters
  defined at the top of this file are used directly.

Phase 3 — Online evaluation (test split):
  The model is retrained from scratch on train + validation data.
  The scaler is re-fit on train + validation features only.
  Then, for each observation in the test split:
    1. Predict t+1 log return using the last LOOKBACK scaled rows.
    2. Record the prediction and the realised return.
    3. Update the model weights via train_on_batch on the new observation.

KEY DIFFERENCES FROM lstm_btc_log_returns_1v2.py
──────────────────────────────────────────────────
  - No random_search() / hyperparameter search of any kind.
  - No PARAM_GRID or N_RANDOM_DRAWS constants.
  - One fixed set of hyperparameters defined explicitly in FIXED_PARAMS.
  - Output CSV files are renamed to reflect the "before" baseline status.
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

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, LSTM
from keras.models import Sequential
from keras.optimizers import Adam


# Reproducibility 
# Fix all random seeds so results are reproducible across runs.
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
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

FORECAST_HORIZON  = 1    # Predict the log return 1 day ahead (t+1)
LOOKBACK          = 100  # Use the last 100 days as input to each prediction
EARLY_STOP_ROUNDS = 10   # EarlyStopping patience in epochs
EARLY_STOP_WINDOW = 30   # Number of trailing sequences held out for early stopping
MAX_EPOCHS        = 100  # Maximum number of training epochs

LAG_FEATURES   = list(range(1, 6))  # Lagged log returns: t-1 to t-5
ROLLING_WINDOW = 5                  # Window size for rolling volatility feature


# Fixed hyperparameters (the "before optimisation" configuration)
# These values are chosen manually as a reasonable starting point.
# No search is performed — this is the configuration to be compared against
# the optimised version.
#
#   lstm_units    : number of memory units in each LSTM layer
#   n_lstm_layers : number of stacked LSTM layers (1 or 2)
#   dropout_rate  : fraction of units dropped for regularisation
#   learning_rate : step size for the Adam optimiser
#   batch_size    : number of samples per gradient update during initial training
FIXED_PARAMS = {
    "lstm_units":    64,
    "n_lstm_layers": 1,
    "dropout_rate":  0.2,
    "learning_rate": 0.001,
    "batch_size":    32,
}











# ── Data loading & feature engineering
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function creates the feature columns used by the LSTM.
# Categorical variables are converted into numeric codes where needed.
# Lagged log returns are added so the model can use recent return history.
# A rolling volatility feature is calculated using previous returns only.
#
# All other numeric columns are shifted by one period to avoid look-ahead bias.
# New columns are first collected in a dictionary and then added at once.
# Rows with NaN values caused by shifting are removed.
# The returned dataframe is ready for sequence construction.
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix from the raw DataFrame.

    All features are lagged by at least one period so that only information
    available strictly before t+1 is used.  This prevents data leakage.

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

    # Lag all other numeric columns by 1 period to avoid look-ahead bias
    skip_cols = (
        {"log_return", f"target_log_return_t_plus_{FORECAST_HORIZON}"}
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











# Load Bitcoin data and create the t+1 prediction target       
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function reads the raw Bitcoin CSV file.
# It sorts rows by date so the time-series order is preserved.
# The closing price column is validated, converted to numeric format,
# and invalid price rows are removed.
#
# Daily log returns are calculated from the closing price series.
# The t+1 target is created by shifting log returns one step backwards.
# Feature engineering is then applied to create lagged model inputs.
# The returned dataframe is cleaned and ready for splitting.
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file, compute log returns and the t+1 target,
    then apply feature engineering.
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

    # The t+1 target is the log return one period ahead of each row
    target_col = f"target_log_return_t_plus_{FORECAST_HORIZON}"
    df[target_col] = df["log_return"].shift(-FORECAST_HORIZON)

    # Drop rows where either the return or the target is NaN
    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    # Apply feature engineering (lags, rolling vol, lagged numeric columns)
    df = build_features(df)

    logging.info(
        "Loaded %d rows with %d features after engineering",
        len(df), len(get_feature_cols(df)),
    )
    return df













# Select the columns used as LSTM input features        
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function returns the final list of model input columns.
# It excludes variables that should not be used as predictors:
# the date, close price, raw log return, target column, and asset identifier.
#
# Any column starting with "target_" is also excluded as extra protection
# against accidental target leakage.
# The resulting feature list is used for scaling, sequence construction,
# training, online evaluation, and prediction.
def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return the ordered list of feature columns used as model inputs.
    Excludes the target column, the date column, the raw price column,
    the raw log return column, and the 'asset' column if present.
    """
    exclude = {
        DATE_COLUMN,
        PRICE_COLUMN,
        "log_return",
        f"target_log_return_t_plus_{FORECAST_HORIZON}",
        "asset",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith("target_")
    ]
    return feature_cols










# Train / validation / test split 
# Split the dataset into train, validation, and test sets       
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function divides the processed dataframe into chronological subsets.
# The split follows the predefined 70/15/15 ratio.
# The training set is used for initial model fitting.
# The validation set is not used for tuning in this baseline version,
# but is later combined with training data for final test evaluation.
#
# The test set is reserved for final out-of-sample evaluation.
# No shuffling is applied because this is time-series forecasting.
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












# Metrics (identical to XGBoost and v1v2 scripts)
# Calculate MAPE safely for small log-return values       
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
# The result is returned as a percentage of correctly predicted directions.
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







# Sequence construction
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function transforms a 2-D feature matrix into 3-D sequences.
# Each sequence contains the previous LOOKBACK observations.
# With LOOKBACK = 100, every prediction uses the last 100 days of inputs.
#
# The target for each sequence is the value directly after the input window.
# This ensures the model only learns from historical information.
# If there are not enough rows to build a sequence, empty arrays are returned.
# The output shape is suitable for Keras LSTM input: samples x timesteps x features.
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









# Build one prediction sequence from the latest context     
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------
# This helper creates a single LSTM input sequence for prediction.
# It takes the last LOOKBACK rows from the scaled feature matrix.
# The sequence is reshaped to include a batch dimension.
#
# For this script, the resulting shape is:
# 1 sample x 100 timesteps x number of features.
# This format is required by Keras when calling model.predict().
# The returned array is converted to float32 for TensorFlow compatibility.
def make_predict_sequence(X_scaled: np.ndarray) -> np.ndarray:
    """
    Return the single prediction sequence: the last LOOKBACK rows of the
    scaled feature matrix, shaped (1, LOOKBACK, F).
    """
    return X_scaled[-LOOKBACK:][np.newaxis, :, :].astype(np.float32)











# LSTM model builder 
# Build and compile the LSTM neural network            
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

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














# Initial training (with early stopping)
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function builds and trains the LSTM using the supplied hyperparameters.
# It trains on pre-built LOOKBACK sequences.
# If enough sequences are available, the last EARLY_STOP_WINDOW sequences are
# held out as a small validation set for early stopping.
#
# Early stopping stops training when validation loss no longer improves,
# which helps reduce overfitting during the initial training phase.
# If too few sequences exist, the model trains on all available sequences.
# The trained model is returned for later online updating.
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
    X_seq  : (N, LOOKBACK, F) sequences built from the training pool
    y_seq  : (N,) corresponding targets
    params : hyperparameter dictionary (must include lstm_units, n_lstm_layers,
             dropout_rate, learning_rate, batch_size)

    Returns Trained Keras Sequential model.
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














# Online evaluation loop
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function simulates a streaming forecasting setup.
# For each new evaluation row, the model first predicts using the latest
# LOOKBACK rows from the context buffer.
# The realised target value is then used to update the model with train_on_batch.
#
# After each step, the new feature row is added to the context buffer.
# This means the model gradually adapts as new observations arrive.
# Predictions and actual values are stored and returned as arrays.
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
      2. Predict t+1 log return.
      3. Update model weights via train_on_batch on the new (sequence, target).
      4. Absorb the new row into the rolling context buffer.

    This simulates a streaming environment where the model adapts after each
    new data point arrives.

    Parameters
    model             : pre-trained Keras LSTM model
    X_eval_scaled     : (M, F) scaled features for the evaluation block
    y_eval            : (M,) targets for the evaluation block
    batch_size        : mini-batch size for train_on_batch
    X_context_scaled  : (T, F) scaled features from the training history
                        preceding the evaluation block (at least LOOKBACK rows)

    Returns
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

        # Step 2: Predict the t+1 log return
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











# Scaler helper
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function fits a StandardScaler on the provided feature matrix.
# It returns both the fitted scaler and the scaled data.
#
# The scaler should only be fitted on training data or train+validation data.
# It must not be fitted on test data, because that would introduce leakage.
# The fitted scaler is later reused to transform validation or test features.
def fit_scaler(X: np.ndarray):
    """
    Fit a StandardScaler on X and return the fitted scaler and the scaled array.

    IMPORTANT: The scaler must always be fit on training data only.
    Fitting on validation or test data would introduce data leakage.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    return scaler, X_scaled










# Model evaluation (train split and test split)
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function performs the main model evaluation.
# For the training split, an initial warm-up section is used to train the model.
# The remaining training rows are evaluated through the online loop.
#
# For the test split, the model is retrained from scratch on train+validation.
# The scaler is also fitted only on train+validation data.
# The test set is then evaluated using predict-then-update online learning.
# The function returns predictions, actual values, and metrics for train and test.
def fit_and_evaluate(
    train_df:     pd.DataFrame,
    valid_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    params:       dict,
    feature_cols: list,
    target_col:   str,
):
    """
    Evaluate the fixed hyperparameter configuration on both the training
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
    """

    # In-sample: online loop over the training split 
    # Use the first (warmup) rows to initialise the model,
    # then run the online loop over the remaining training rows.
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

    # Out-of-sample: retrain on train + validation, then online over test 
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

    return (test_pred, test_actual, test_metrics,
            train_pred, train_actual, train_metrics)














# Output
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This function exports the final test predictions.
# It aligns the prediction rows with the original dataframe’s test section.
# Actual t+1 log returns and predicted t+1 log returns are added as columns.
#
# The fixed lookback window and fixed baseline hyperparameters are also saved.
# This makes the output traceable and comparable with optimised versions.
# No hyperparameter search file is saved because this is the before-tuning model.
# The CSV can later be used for plotting, reporting, and model comparison.
def save_outputs(
    full_df:     pd.DataFrame,
    train_len:   int,
    valid_len:   int,
    test_pred:   np.ndarray,
    test_actual: np.ndarray,
    params:      dict,
    target_col:  str,
) -> None:
    """
    Save the test predictions to a CSV file.

    The output format mirrors the original script (lstm_btc_log_returns_1v2.py)
    so that results can be compared directly.  The file is renamed to reflect
    that this is the "before hyperparameter optimisation" baseline.

    Unlike the v1v2 script, no grid search results CSV is saved here
    because no search was performed.
    """
    # The test predictions start immediately after the train + validation rows
    start_idx = train_len + valid_len
    pred_df   = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()

    # Add prediction columns
    pred_df["actual_log_return_t_plus_1"]    = test_actual
    pred_df["predicted_log_return_t_plus_1"] = test_pred

    # Record the fixed settings used in this run for traceability
    pred_df["lookback_window"] = LOOKBACK
    for key, value in params.items():
        pred_df[f"best_{key}"] = value

    # Save to CSV
    output_filename = "lstm_test_predictions_before_hyper_1.csv"
    pred_df.to_csv(output_filename, index=False)

    logging.info("Saved → %s", output_filename)











# Main
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------

# This is the main controller function for the script.
# It loads the data, creates features, selects model inputs, and splits the data.
# It then evaluates the fixed LSTM configuration without hyperparameter search.
#
# Results are calculated for both the training split and the final test split.
# The metrics are logged and printed in a formatted terminal output.
# Finally, the test predictions and model settings are saved to CSV.
# This function is executed when the script is run directly.
def lstm_forecast() -> None:
    """
    Main entry point.  Orchestrates the full pipeline:
      1. Load and preprocess data
      2. Split into train / validation / test
      3. Evaluate the fixed hyperparameter set on train and test splits
      4. Print results to the terminal
      5. Save predictions to CSV
    """

    # Step 1: Load data, compute log returns, build features
    df           = load_data(CSV_PATH)
    feature_cols = get_feature_cols(df)
    target_col   = f"target_log_return_t_plus_{FORECAST_HORIZON}"

    # Step 2: Chronological split
    train_df, valid_df, test_df = split_data(df)

    # Step 3: Log which hyperparameters will be used
    logging.info(
        "Fixed hyperparameters (no search): %s  |  lookback=%d (fixed)",
        FIXED_PARAMS, LOOKBACK,
    )

    # Step 4: Train the model and evaluate on both train and test splits
    (test_pred, test_actual, test_metrics,
     train_pred, train_actual, train_metrics) = fit_and_evaluate(
        train_df     = train_df,
        valid_df     = valid_df,
        test_df      = test_df,
        params       = FIXED_PARAMS,
        feature_cols = feature_cols,
        target_col   = target_col,
    )

    # Step 5: Log results to the terminal (same format as v1v2)
    logging.info("----- TRAIN RESULTS (online learning, t+%d) -----", FORECAST_HORIZON)
    for metric_name, metric_value in train_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info("----- FINAL TEST RESULTS (online learning, t+%d) -----", FORECAST_HORIZON)
    for metric_name, metric_value in test_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    # Step 6: Print a formatted results block to the console (same format as v1v2)
    print("\n" + "=" * 60)
    print("  TRAIN RESULTS  —  LSTM Online Learning (t+1 forecast)")
    print("=" * 60)
    print(f"  Fixed params            : {FIXED_PARAMS}")
    print(f"  Lookback window         : {LOOKBACK} (fixed)")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {train_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  FINAL TEST RESULTS  —  LSTM Online Learning (t+1 forecast)")
    print("=" * 60)
    print(f"  Fixed params            : {FIXED_PARAMS}")
    print(f"  Lookback window         : {LOOKBACK} (fixed)")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    # Step 7: Save test predictions to CSV
    save_outputs(
        full_df     = df,
        train_len   = len(train_df),
        valid_len   = len(valid_df),
        test_pred   = test_pred,
        test_actual = test_actual,
        params      = FIXED_PARAMS,
        target_col  = target_col,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
    )
    lstm_forecast()
