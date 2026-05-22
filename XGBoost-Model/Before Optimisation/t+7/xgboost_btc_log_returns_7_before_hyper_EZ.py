# xgboost_btc_log_returns_7_before_hyper_EZ.py
#
# XGBoost model to predict Bitcoin log returns at t+7.
#
# VERSION: Before hyperparameter optimisation (fixed-hyperparameter baseline)
#
# This is the "before" version. It uses one manually chosen set of hyperparameters
# and does NOT run any kind of hyperparameter search. The goal is to have a simple
# baseline to compare against the optimised version later.
#
# The hyperparameters below are just sensible defaults, not tuned to this dataset:
#   n_estimators    = 100  (number of trees)
#   max_depth       = 5    (how deep each tree is)
#   learning_rate   = 0.1  (how fast the model learns)
#   subsample       = 0.8  (fraction of rows used per tree)
#   colsample_bytree= 0.8  (fraction of columns used per tree)
#
# The evaluation setup is identical to the optimised version so results are
# directly comparable:
#   - 70/15/15 train/validation/test split
#   - Walk-forward validation on the validation set
#   - Rolling out-of-sample evaluation on the test set
#   - Same metrics: MAE, RMSE, MAPE, Directional Accuracy
#   - Same CSV output format
#
# DIFFERENCE vs t+1 version:
#   - FORECAST_HORIZON = 7 (predict 7 days ahead instead of 1)
#   - LAG_FEATURES extended to [1,2,3,4,5,6,7,14] so the model can see
#     enough history to forecast 7 steps ahead without data leakage
#   - ROLLING_WINDOW increased to 7 to match the forecast horizon
#   - All output filenames and column names updated to reflect t+7

import logging
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==============================================================================
# SETTINGS
# ==============================================================================

CSV_PATH      = "btc_clean.csv"
PRICE_COLUMN  = "Close"
DATE_COLUMN   = "Date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

FORECAST_HORIZON   = 7   # predict t+7 log return
RETRAIN_STRIDE     = 1   # retrain at every step during evaluation
EARLY_STOP_ROUNDS  = 20  # stop early if no improvement after this many rounds
EARLY_STOP_WINDOW  = 30  # how many recent rows to use as the early-stopping eval set

# Lags extended vs the t+1 version: we now include lags up to 14 days
# so the model has enough historical signal to predict 7 days ahead.
LAG_FEATURES   = [1, 2, 3, 4, 5, 6, 7, 14]
ROLLING_WINDOW = 7   # rolling volatility window matches the forecast horizon

RANDOM_SEED = 42

# Fixed hyperparameters — same defaults as the t+1 version, no tuning done here
FIXED_TUNED_PARAMS = {
    "n_estimators"    : 100,
    "max_depth"       : 5,
    "learning_rate"   : 0.1,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
}

# XGBoost settings that never change
FIXED_PARAMS = {
    "objective"   : "reg:squarederror",
    "eval_metric" : "rmse",
    "tree_method" : "hist",
    "random_state": RANDOM_SEED,
    "verbosity"   : 0,
}











# ==========================================================
# STEP 1 — LOAD DATA                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================
# This function reads the raw Bitcoin dataset from CSV and prepares the
# basic target variable for forecasting.
#
# It first sorts the observations by date to preserve the time-series order.
# The closing price column is then converted to numeric format and invalid
# price rows are removed.
#
# Daily log returns are calculated from the closing price. The target column
# is created by shifting the log return series 7 days backwards, so each row
# contains the future t+7 return that the model should predict.
#
# Rows with missing log returns or missing target values are removed before
# the cleaned dataframe is returned.
def load_data(csv_path):
    """
    Read the CSV, calculate daily log returns, and create the target column.
    The target is the log return FORECAST_HORIZON days in the future.
    """
    logging.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sort by date so rows are in chronological order
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Make sure the price column exists and is numeric
    if PRICE_COLUMN not in df.columns:
        raise ValueError("Column '" + PRICE_COLUMN + "' not found in " + csv_path)
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
    df = df.dropna(subset=[PRICE_COLUMN]).reset_index(drop=True)

    # Calculate daily log return: log(today's close / yesterday's close)
    df["log_return"] = np.log(df[PRICE_COLUMN]).diff()

    # The target is the log return 7 days from now
    # shift(-7) moves the value 7 rows up so it aligns with the current row
    target_col = "target_log_return_t_plus_7"
    df[target_col] = df["log_return"].shift(-FORECAST_HORIZON)

    # Drop rows where we couldn't compute a return or a target
    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    logging.info("Rows after loading: %d", len(df))
    return df














# =====================================================================
# STEP 2 — BUILD FEATURES                        (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==========================================================
# This function creates the predictive input variables used by the XGBoost model.
# All features are constructed using past information only, which helps prevent
# data leakage from future observations.
#
# Text-based columns are converted into numeric category codes because XGBoost
# requires numerical inputs.
#
# The function adds lagged log returns, including lags up to 14 days, so the
# model has access to historical return patterns relevant for a 7-day forecast.
# It also adds 7-day rolling volatility, shifted by one day to avoid using
# current or future information.
#
# Other numeric variables are lagged by one day, and rows with missing values
# caused by shifting are removed.
def build_features(df):
    """
    Add new columns to the dataframe that the model will use as inputs.
    All features are lagged so we never accidentally use future data (data leakage).
    """
    df = df.copy()

    # XGBoost needs numbers. Convert any text columns (like 'bull'/'bear') to integers.
    text_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in text_columns:
        if col not in [DATE_COLUMN, "asset"]:
            df[col] = pd.Categorical(df[col]).codes

    # --- Lagged log returns ---
    # lag_1 = yesterday's return, lag_2 = two days ago, etc.
    # These are the most direct historical signals for the model.
    for lag in LAG_FEATURES:
        col_name = "lag_" + str(lag)
        df[col_name] = df["log_return"].shift(lag)

    # --- Rolling volatility ---
    # Standard deviation of the last 7 returns (shifted by 1 to avoid leakage).
    # This tells the model how volatile the market has been recently.
    df["rolling_vol_7"] = df["log_return"].shift(1).rolling(window=ROLLING_WINDOW).std()

    # --- Lag all other numeric columns by 1 day ---
    # Any other numeric columns in the CSV (e.g. volume, RSI, etc.) are also
    # shifted by 1 so we only use yesterday's value, not today's.
    already_created = (
        ["log_return", "target_log_return_t_plus_7", "rolling_vol_7"]
        + ["lag_" + str(l) for l in LAG_FEATURES]
    )
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in already_created:
            df[col + "_lag1"] = df[col].shift(1)

    # Drop any rows that have NaN (they appear at the start due to the shifting)
    df = df.dropna().reset_index(drop=True)

    logging.info("Rows after feature engineering: %d", len(df))
    return df













# ==============================================================================
# STEP 3 — IDENTIFY FEATURE COLUMNS                        (IMPROVED BY CLAUDE-Sonnet-4.6)
# =====================================================
# This function identifies which columns should be used as model inputs.
#
# Columns that should not be used for prediction are excluded, including:
# the date, raw closing price, raw log return, target variable, and asset
# identifier column.
#
# Any column starting with "target_" is also excluded to avoid accidentally
# using the prediction target as an input feature.
#
# The returned feature list is used consistently during training, validation,
# testing, and walk-forward prediction.
def get_feature_cols(df):
    """
    Return a list of column names that the model should use as inputs.
    We exclude the target, the date, the raw price, and the raw log return.
    """
    columns_to_exclude = [
        DATE_COLUMN,
        PRICE_COLUMN,
        "log_return",
        "target_log_return_t_plus_7",
        "asset",
    ]

    feature_cols = []
    for col in df.columns:
        # Also skip any other target columns that might exist
        if col not in columns_to_exclude and not col.startswith("target_"):
            feature_cols.append(col)

    return feature_cols














# ==============================================================================
# STEP 4 — SPLIT INTO TRAIN / VALIDATION / TEST                 (IMPROVED BY CLAUDE-Sonnet-4.6)
# =======================================================
# This function divides the fully prepared dataset into three time-ordered
# subsets using the predefined 70/15/15 split.
#
# The training set is used to fit the model, the validation set is used to
# assess the fixed baseline configuration before testing, and the test set is
# reserved for the final out-of-sample evaluation.
#
# No random shuffling is applied because this is a time-series forecasting task.
# Keeping chronological order ensures the model only learns from observations
# that would have been available at that point in time.
def split_data(df):
    """
    Split the data into three sequential chunks (no shuffling — time series!).
      - Train (70%): used to train the model
      - Validation (15%): used to check performance before touching the test set
      - Test (15%): final out-of-sample evaluation, only used once at the end
    """
    total_rows = len(df)
    train_end  = int(total_rows * TRAIN_RATIO)
    valid_end  = train_end + int(total_rows * VALID_RATIO)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df  = df.iloc[valid_end:].copy()

    logging.info("Split sizes — Train: %d  |  Valid: %d  |  Test: %d",
                 len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df















# ==============================================================================
# STEP 5 — METRICS                           (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Calculate MAPE with protection against near-zero values
# ==============================================================================
# This function computes the Mean Absolute Percentage Error between actual
# and predicted log returns.
#
# Since Bitcoin log returns can be very close to zero, dividing by the actual
# value may produce unstable or extremely large percentage errors.
#
# To avoid this, the denominator is clipped using a small epsilon value.
# This makes the MAPE calculation numerically safer while still preserving
# the general interpretation of percentage error.
def safe_mape(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    We clip the denominator to epsilon so we never divide by zero.
    Log returns are often very small, so this protection matters here.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.maximum(np.abs(y_true), epsilon)
    percentage_errors = np.abs((y_true - y_pred) / denominator) * 100.0
    return float(np.mean(percentage_errors))





# Calculate directional accuracy             (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function measures whether the model correctly predicts the direction
# of the future Bitcoin return.
#
# It compares the sign of the true t+7 log return with the sign of the
# predicted t+7 log return.
#
# A prediction is counted as correct when both values are positive or both
# values are negative. The final score is returned as a percentage.
#
# This metric is useful because, in financial forecasting, predicting the
# correct direction can still be informative even when exact values differ.
def directional_accuracy(y_true, y_pred):
    """
    What fraction of the time did we predict the correct direction?
    (i.e. both positive, or both negative)
    Reported as a percentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct_direction) * 100.0)









# Calculate all evaluation metrics                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function computes the main metrics used to evaluate the XGBoost
# forecasting performance.
#
# MAE measures the average absolute forecast error, while RMSE penalises
# larger errors more strongly.
#
# MAPE provides a percentage-based error measure using the safe MAPE helper.
# Directional Accuracy measures whether the model predicts the correct return
# direction.
#
# The metrics are returned as a dictionary so they can easily be logged,
# printed, saved, or compared with other models.
def calculate_metrics(y_true, y_pred):
    """Compute all four evaluation metrics and return them as a dictionary."""
    return {
        "MAE"                 : mean_absolute_error(y_true, y_pred),
        "RMSE"                : sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE"                : safe_mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }












# ==============================================================================
# STEP 6 — TRAIN A SINGLE XGBOOST MODEL                     (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function fits an XGBoost model using the supplied training data and
# fixed baseline hyperparameters.
#
# When enough observations are available, the last part of the training data
# is used as a small evaluation set for early stopping.
#
# Early stopping prevents the model from continuing to add trees when the
# evaluation performance no longer improves, which can reduce overfitting.
#
# If the training set is too small for early stopping, the model is fitted
# normally on all available observations.
def train_xgb(X_train, y_train, params):
    """
    Train one XGBoost model on the given data.

    We use early stopping to prevent overfitting:
      - Hold out the last EARLY_STOP_WINDOW rows as a small validation set.
      - If the model doesn't improve on that set for EARLY_STOP_ROUNDS rounds, stop.
    If there isn't enough data for early stopping, we just train normally.
    """
    not_enough_data = len(X_train) <= EARLY_STOP_WINDOW + 1

    if not_enough_data:
        # Not enough rows to hold out a validation set, so skip early stopping
        model = xgb.XGBRegressor(**params, **FIXED_PARAMS)
        model.fit(X_train, y_train)
        return model

    # Split training data: most rows for fitting, last rows for early-stop check
    X_fit  = X_train[:-EARLY_STOP_WINDOW]
    y_fit  = y_train[:-EARLY_STOP_WINDOW]
    X_eval = X_train[-EARLY_STOP_WINDOW:]
    y_eval = y_train[-EARLY_STOP_WINDOW:]

    model = xgb.XGBRegressor(**params, **FIXED_PARAMS,
                              early_stopping_rounds=EARLY_STOP_ROUNDS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)

    return model










# ==============================================================================
# STEP 7 — WALK-FORWARD EVALUATION                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function evaluates the model in a realistic time-series setting.
#
# It moves through the evaluation set one observation at a time. At each step,
# the model is trained on all data available up to that point and then predicts
# the next unseen observation.
#
# After the prediction is made, the true observation is added to the training
# pool so it can be used in later steps.
#
# With stride=1, the model is retrained at every step, giving the most detailed
# rolling out-of-sample evaluation, although it is computationally slower.
def walk_forward(train_df, eval_df, params, feature_cols, target_col, stride=1):
    """
    Simulate how the model would be used in real life.

    We step through the evaluation set one row at a time:
      1. Train the model on everything seen so far.
      2. Predict the next row.
      3. Add that row to the training pool.
      4. Repeat.

    This is called "walk-forward" because the training window walks forward
    through time. stride=1 means we retrain at every single step.
    """
    predictions = []
    actuals     = []

    # Start with the training data; this pool will grow as we step forward
    training_pool = train_df.copy()
    model = None

    # Reset index so we can iterate cleanly with .iloc
    eval_rows = eval_df.reset_index(drop=True)

    for i in range(len(eval_rows)):
        current_row = eval_rows.iloc[[i]]

        # Retrain the model at the first step, and then every `stride` steps
        retrain_now = (model is None) or (i % stride == 0)
        if retrain_now:
            X_pool = training_pool[feature_cols].values
            y_pool = training_pool[target_col].values
            model  = train_xgb(X_pool, y_pool, params)

        # Predict the current row and record both the prediction and the truth
        X_current  = current_row[feature_cols].values
        prediction = float(model.predict(X_current)[0])
        actual     = float(current_row[target_col].values[0])

        predictions.append(prediction)
        actuals.append(actual)

        # Add this row to the pool so future training steps include it
        training_pool = pd.concat([training_pool, current_row], ignore_index=True)

    return np.array(predictions), np.array(actuals)














# ==============================================================================
# STEP 8 — EVALUATE ON VALIDATION SET                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Evaluate the fixed XGBoost baseline on the validation set
# ==============================================================================
# This function applies walk-forward forecasting to the validation set using
# the manually selected fixed hyperparameters.
#
# In this before-optimisation version, no hyperparameter search is performed.
# The validation set is therefore used only to assess how the baseline
# configuration performs before final testing.
#
# The function returns the validation predictions, actual values, and computed
# evaluation metrics.
def evaluate_on_validation(train_df, valid_df, feature_cols, target_col):
    """
    Run walk-forward evaluation on the validation set using the fixed hyperparameters.
    This is where we would normally do a hyperparameter search, but in this
    "before" version we just use the fixed defaults.
    """
    logging.info("Running walk-forward on validation set with fixed params: %s",
                 FIXED_TUNED_PARAMS)

    predictions, actuals = walk_forward(
        train_df, valid_df,
        FIXED_TUNED_PARAMS,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )

    metrics = calculate_metrics(actuals, predictions)

    logging.info("Validation — RMSE=%.8f | MAE=%.8f | MAPE=%.4f | DA=%.2f%%",
                 metrics["RMSE"], metrics["MAE"], metrics["MAPE"],
                 metrics["Directional_Accuracy"])

    return predictions, actuals, metrics















# ==============================================================================
# STEP 9 — EVALUATE ON TRAIN AND TEST SETS                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Evaluate the fixed XGBoost model on train and test sets
# ==============================================================================
# This function calculates both in-sample and out-of-sample performance for
# the baseline XGBoost model.
#
# For the training evaluation, an initial warm-up section is used so the model
# has enough historical data before predictions begin.
#
# For the test evaluation, the train and validation sets are combined as the
# starting training pool before walking forward through the held-out test set.
#
# The function returns predictions, actual values, and metrics for both train
# and test evaluation.
def fit_and_evaluate(train_df, valid_df, test_df, feature_cols, target_col):
    """
    Evaluate on both the training set (in-sample) and the test set (out-of-sample).

    For training: we skip the first few rows as a warm-up window, then walk forward
    through the rest. This ensures the model always has enough history to start.

    For test: we combine train + validation as the starting training pool,
    then walk forward through the test set.
    """

    # --- Train evaluation ---
    # We need a warm-up window so the model has enough data from the start
    warmup_size  = EARLY_STOP_WINDOW + max(LAG_FEATURES) + ROLLING_WINDOW
    train_warmup = train_df.iloc[:warmup_size].copy()   # seed data, not evaluated
    train_eval   = train_df.iloc[warmup_size:].copy()   # the part we evaluate on

    train_predictions, train_actuals = walk_forward(
        train_warmup, train_eval,
        FIXED_TUNED_PARAMS,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )
    train_metrics = calculate_metrics(train_actuals, train_predictions)

    # --- Test evaluation ---
    # Use all of train + validation as the starting pool before stepping through test
    train_and_valid = pd.concat([train_df, valid_df], ignore_index=True)

    test_predictions, test_actuals = walk_forward(
        train_and_valid, test_df,
        FIXED_TUNED_PARAMS,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )
    test_metrics = calculate_metrics(test_actuals, test_predictions)

    return (test_predictions, test_actuals, test_metrics,
            train_predictions, train_actuals, train_metrics)














# ==============================================================================
# STEP 10 — SAVE RESULTS TO CSV                     (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function exports the key outputs from the baseline XGBoost experiment.
#
# First, it saves the fixed hyperparameter configuration together with the
# validation metrics. This provides a record of the baseline setup.
#
# Second, it saves the final test predictions alongside the actual t+7 log
# returns, so the results can be inspected, plotted, or compared later.
#
# The selected hyperparameters are also added as columns for traceability.
def save_outputs(full_df, train_len, valid_len,
                 test_pred, test_actual,
                 fixed_params, valid_metrics, target_col):
    """
    Save two CSV files:
      1. The fixed hyperparameter config and its validation metrics.
      2. The test set predictions alongside the actual values.
    """

    # --- File 1: hyperparameter config + validation metrics ---
    validation_result_row = dict(fixed_params)
    validation_result_row["validation_rmse"]                 = valid_metrics["RMSE"]
    validation_result_row["validation_mae"]                  = valid_metrics["MAE"]
    validation_result_row["validation_mape"]                 = valid_metrics["MAPE"]
    validation_result_row["validation_directional_accuracy"] = valid_metrics["Directional_Accuracy"]

    validation_results_df = pd.DataFrame([validation_result_row])
    validation_results_df.to_csv("xgboost_fixed_params_results_7.csv", index=False)

    # --- File 2: test predictions ---
    start_idx = train_len + valid_len
    pred_df   = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()

    pred_df["actual_log_return_t_plus_7"]    = test_actual
    pred_df["predicted_log_return_t_plus_7"] = test_pred

    # Also save the hyperparameter values as columns (prefix "best_" for compatibility)
    for param_name, param_value in fixed_params.items():
        pred_df["best_" + param_name] = param_value

    pred_df.to_csv("xgboost_test_predictions_7.csv", index=False)

    logging.info("Saved -> xgboost_fixed_params_results_7.csv")
    logging.info("Saved -> xgboost_test_predictions_7.csv")













# ==============================================================================
# MAIN                           (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function controls the full workflow of the before-optimisation model.
#
# It loads and prepares the Bitcoin data, creates t+7 forecasting features,
# identifies feature columns, and splits the dataset chronologically.
#
# It then evaluates the fixed XGBoost hyperparameters on validation, training,
# and test data using walk-forward forecasting.
#
# Finally, it logs and prints the results and saves the output CSV files.
# This function is the main entry point for the full experiment.
def xgboost_forecast():
    """Run the full pipeline from loading data to saving results."""

    # --- Load and prepare data ---
    df = load_data(CSV_PATH)
    df = build_features(df)
    feature_cols = get_feature_cols(df)
    target_col   = "target_log_return_t_plus_7"

    logging.info("Number of feature columns: %d", len(feature_cols))

    # --- Split into train / validation / test ---
    train_df, valid_df, test_df = split_data(df)

    # --- Evaluate fixed hyperparameters on validation set ---
    valid_pred, valid_actual, valid_metrics = evaluate_on_validation(
        train_df, valid_df, feature_cols, target_col
    )

    # --- Evaluate on train and test sets ---
    (test_pred, test_actual, test_metrics,
     train_pred, train_actual, train_metrics) = fit_and_evaluate(
        train_df, valid_df, test_df, feature_cols, target_col
    )

    # --- Log all results ---
    logging.info("----- TRAIN RESULTS (walk-forward, t+7) -----")
    for metric_name, metric_value in train_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info("----- VALIDATION RESULTS (walk-forward, t+7) -----")
    for metric_name, metric_value in valid_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info("----- FINAL TEST RESULTS (walk-forward, t+7) -----")
    for metric_name, metric_value in test_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    # --- Print formatted summaries to the terminal ---
    print("\n" + "=" * 56)
    print("  TRAIN RESULTS  -  XGBoost (t+7 forecast)")
    print("  [Before Hyperparameter Optimisation - Fixed Params]")
    print("=" * 56)
    print(f"  Fixed params            : {FIXED_TUNED_PARAMS}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {train_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 56)

    print("\n" + "=" * 56)
    print("  VALIDATION RESULTS  -  XGBoost (t+7 forecast)")
    print("  [Before Hyperparameter Optimisation - Fixed Params]")
    print("=" * 56)
    print(f"  Fixed params            : {FIXED_TUNED_PARAMS}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {valid_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {valid_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {valid_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {valid_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 56)

    print("\n" + "=" * 56)
    print("  FINAL TEST RESULTS  -  XGBoost (t+7 forecast)")
    print("  [Before Hyperparameter Optimisation - Fixed Params]")
    print("=" * 56)
    print(f"  Fixed params            : {FIXED_TUNED_PARAMS}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 56)

    # --- Save CSV outputs ---
    save_outputs(
        full_df      = df,
        train_len    = len(train_df),
        valid_len    = len(valid_df),
        test_pred    = test_pred,
        test_actual  = test_actual,
        fixed_params = FIXED_TUNED_PARAMS,
        valid_metrics= valid_metrics,
        target_col   = target_col,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    xgboost_forecast()
