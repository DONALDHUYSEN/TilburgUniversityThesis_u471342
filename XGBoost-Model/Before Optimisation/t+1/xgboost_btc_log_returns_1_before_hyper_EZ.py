# xgboost_btc_log_returns_1_before_hyper.py
#
# XGBoost model to predict Bitcoin log returns at t+1.
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

import logging
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --- Settings ---

CSV_PATH = "btc_clean.csv"
PRICE_COLUMN = "Close"
DATE_COLUMN = "Date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

FORECAST_HORIZON = 1   # predict t+1 log return
RETRAIN_STRIDE = 1     # retrain at every step during evaluation
EARLY_STOP_ROUNDS = 20 # stop early if no improvement after this many rounds
EARLY_STOP_WINDOW = 30 # how many recent rows to use as the early-stopping eval set

LAG_FEATURES = [1, 2, 3, 4, 5]  # use log returns from t-1 to t-5 as features
ROLLING_WINDOW = 5               # window size for rolling volatility feature

RANDOM_SEED = 42

# These are the fixed hyperparameters for this "before" version.
# No search is done - we just use these values as-is.
FIXED_TUNED_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# These XGBoost settings are always fixed and not part of the tuning
FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}










# ── Data loading & feature engineering ─────────────────(IMPROVED BY CLAUDE-Sonnet-4.6)
# This function generates lagged and rolling-window features using only
# past information in order to prevent data leakage into the future.
#
# The feature engineering process includes:
# - lagged log returns (t-1 to t-5)
# - rolling volatility based on previous returns
# - lagged versions of other numeric variables
# - label encoding of categorical variables for XGBoost compatibility
#
# Rows containing missing values caused by shifting operations are removed
# before returning the final feature-enhanced dataframe.
# --- Feature engineering ---
def build_features(df):
    # We need to create features that only use past data (no leakage into the future)
    df = df.copy()

    # Label-encode any text columns like Market_Regime (e.g. 'bull'/'bear' -> 0/1)
    # XGBoost needs numbers, not strings
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        if col not in [DATE_COLUMN, "asset"]:
            df[col] = pd.Categorical(df[col]).codes

    # Build new feature columns and add them all at once at the end
    new_cols = {}

    # Add lagged log returns as features (lag_1 = yesterday's return, etc.)
    for lag in LAG_FEATURES:
        new_cols["lag_" + str(lag)] = df["log_return"].shift(lag)

    # Add rolling volatility (standard deviation of last 5 returns, shifted to avoid leakage)
    new_cols["rolling_vol_5"] = df["log_return"].shift(1).rolling(window=ROLLING_WINDOW).std()

    # Also lag all other numeric columns by 1 so we only use info available at time t
    skip_cols = set(["log_return", "target_log_return_t_plus_" + str(FORECAST_HORIZON),
                     "rolling_vol_5"] + ["lag_" + str(l) for l in LAG_FEATURES])
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in skip_cols:
            new_cols[col + "_lag1"] = df[col].shift(1)

    # Add all new columns to the dataframe in one go
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.dropna().reset_index(drop=True)
    return df









# Load and preprocess Bitcoin price data for forecasting      (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function:
# - loads the CSV dataset
# - sorts observations chronologically
# - computes daily Bitcoin log returns
# - creates the prediction target (t+1 log return)
# - applies feature engineering to generate model inputs
#
# The returned dataframe contains cleaned data, engineered features,
# and prediction targets ready for XGBoost forecasting.
def load_data(csv_path):
    logging.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sort by date if the column exists
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if PRICE_COLUMN not in df.columns:
        raise ValueError("Column '" + PRICE_COLUMN + "' not found in " + csv_path)

    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
    df = df.dropna(subset=[PRICE_COLUMN]).reset_index(drop=True)

    # Calculate daily log returns: log(today / yesterday)
    df["log_return"] = np.log(df[PRICE_COLUMN]).diff()

    # The target is tomorrow's log return
    target_col = "target_log_return_t_plus_" + str(FORECAST_HORIZON)
    df[target_col] = df["log_return"].shift(-FORECAST_HORIZON)

    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    # Build all the features
    df = build_features(df)

    logging.info("Loaded %d rows with %d features after engineering", len(df), len(get_feature_cols(df)))
    return df










# Select all usable feature columns for model training             (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function removes columns that should not be used as model inputs,
# such as:
# - target variables
# - dates
# - raw prices
# - identifier columns
#
# The remaining columns are returned as the final feature set used
# by the XGBoost model.
def get_feature_cols(df):
    # Return all columns that are actual features (not the target, date, price, etc.)
    exclude = [DATE_COLUMN, PRICE_COLUMN, "log_return",
               "target_log_return_t_plus_" + str(FORECAST_HORIZON), "asset"]
    feature_cols = []
    for col in df.columns:
        if col not in exclude and not col.startswith("target_"):
            feature_cols.append(col)
    return feature_cols















# Split the dataset into train, validation, and test subsets             (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# The dataset is divided chronologically using a 70/15/15 split:
# - training set    : model fitting
# - validation set  : model evaluation before testing
# - test set        : final out-of-sample evaluation
#
# No random shuffling is applied because time-series forecasting
# must preserve temporal order.
# --------------------------------
# --- Train/validation/test split ---

def split_data(df):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    logging.info("Train: %d  |  Valid: %d  |  Test: %d", len(train), len(valid), len(test))
    return train, valid, test









# Compute MAPE while safely handling near-zero actual values              (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# Bitcoin log returns can be extremely close to zero, which may cause
# instability in percentage-based error calculations.
#
# A small epsilon value is therefore used as a lower denominator bound
# to prevent division errors and unrealistic MAPE values.
# ── Metrics ─────────────────────────────────────────────────────
# --- Metrics ---

def safe_mape(y_true, y_pred, epsilon=1e-8):
    # MAPE can blow up when actual values are near zero (common with log returns)
    # so we clip the denominator to a small number to avoid that
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)










# Calculate directional forecasting accuracy                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This metric measures how often the predicted return direction
# matches the true market movement direction.
#
# A prediction is considered correct if both the predicted and actual
# log returns have the same sign (positive or negative).
def directional_accuracy(y_true, y_pred):
    # What percentage of the time did we predict the right direction (up or down)?
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0)







# Compute the forecasting evaluation metrics                            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function calculates the main performance metrics used to assess
# XGBoost forecasting accuracy:
# - MAE
# - RMSE
# - MAPE
# - Directional Accuracy
#
# The metrics are returned as a dictionary for easier reporting
# and result storage.
def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }








# Train the XGBoost forecasting model                              (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function fits an XGBoost regression model using the provided
# training data and hyperparameters.
#
# Early stopping is applied when sufficient data is available:
# - the most recent observations are used as a temporary evaluation set
# - training stops automatically if performance no longer improves
#
# This helps reduce overfitting and improves generalisation performance.
# ── XGBoost helpers ────────────────────────────────────────────────
# --- XGBoost training ---

def train_xgb(X_train, y_train, params):
    # Train an XGBoost model. We use early stopping to avoid overfitting:
    # hold out the last EARLY_STOP_WINDOW rows as a small eval set and stop
    # if performance on it doesn't improve for EARLY_STOP_ROUNDS rounds.

    if len(X_train) <= EARLY_STOP_WINDOW + 1:
        # Not enough data to use early stopping, so just fit normally
        model = xgb.XGBRegressor(**params, **FIXED_PARAMS)
        model.fit(X_train, y_train)
        return model

    # Split off the last few rows as the early-stopping eval set
    X_fit = X_train[:-EARLY_STOP_WINDOW]
    y_fit = y_train[:-EARLY_STOP_WINDOW]
    X_eval = X_train[-EARLY_STOP_WINDOW:]
    y_eval = y_train[-EARLY_STOP_WINDOW:]

    model = xgb.XGBRegressor(**params, **FIXED_PARAMS, early_stopping_rounds=EARLY_STOP_ROUNDS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)

    return model









# Perform rolling walk-forward forecasting                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function simulates real-world forecasting by moving through the
# evaluation dataset one timestep at a time.
#
# At each step:
# - the model is trained on all available historical observations
# - a prediction is generated for the next unseen observation
# - the true observation is added to the training pool
#
# This creates a realistic rolling out-of-sample evaluation framework.
# --- Walk-forward evaluation ---

def walk_forward(train_df, eval_df, params, feature_cols, target_col, stride=1):
    # Walk-forward means we move through the eval set one row at a time.
    # At each step we retrain on everything seen so far, then predict the next row.
    # This simulates how the model would actually be used in practice.
    # stride=1 means we retrain at every single step (most accurate but slowest).

    predictions = []
    actuals = []

    pool_df = train_df.copy()  # this grows as we step through the eval set
    model = None

    eval_values = eval_df.reset_index(drop=True)

    for i in range(len(eval_values)):
        row = eval_values.iloc[[i]]

        # Retrain if this is the first step, or every `stride` steps
        if model is None or i % stride == 0:
            X_pool = pool_df[feature_cols].values
            y_pool = pool_df[target_col].values
            model = train_xgb(X_pool, y_pool, params)

        # Make a prediction for this row
        X_pred = row[feature_cols].values
        pred_value = float(model.predict(X_pred)[0])
        predictions.append(pred_value)
        actuals.append(float(row[target_col].values[0]))

        # Add this row to the training pool for future steps
        pool_df = pd.concat([pool_df, row], ignore_index=True)

    return np.array(predictions), np.array(actuals)













# ---------------------------------------------------------------------------
# Evaluate the fixed XGBoost hyperparameters on the validation set            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function applies walk-forward forecasting on the validation set
# using the manually selected baseline hyperparameters.
#
# The resulting predictions are used to calculate forecasting metrics
# and assess baseline model performance before optimisation.
# ── Validation evaluation (replaces random_search in the optimised version) ───
# --- Validation evaluation ---

def evaluate_on_validation(train_df, valid_df, feature_cols, target_col):
    # Evaluate the fixed hyperparameters on the validation set.
    # This replaces the random search from the optimised version.
    # We just run walk-forward with our one fixed set of params.

    logging.info("Evaluating fixed hyperparameters on validation set: %s", FIXED_TUNED_PARAMS)

    pred, actual = walk_forward(train_df, valid_df, FIXED_TUNED_PARAMS, feature_cols, target_col, stride=RETRAIN_STRIDE)
    metrics = calculate_metrics(actual, pred)

    logging.info("Validation — RMSE=%.8f | MAE=%.8f | MAPE=%.4f | DA=%.2f%%",
                 metrics["RMSE"], metrics["MAE"], metrics["MAPE"], metrics["Directional_Accuracy"])

    return pred, actual, metrics













# Evaluate XGBoost performance on the training and test datasets               (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function performs:
# - in-sample evaluation on the training set
# - out-of-sample evaluation on the test set
#
# Both evaluations use the rolling walk-forward methodology so that
# forecasting conditions remain consistent across datasets.
# --- Train and test evaluation ---

def fit_and_evaluate(train_df, valid_df, test_df, feature_cols, target_col):
    # Evaluate performance on both the train set (in-sample) and test set (out-of-sample).
    # Both use stride=1 so the model is retrained at every step.

    # For the train evaluation, we need a small warm-up window first so the model
    # has enough history to start making predictions
    warmup = EARLY_STOP_WINDOW + max(LAG_FEATURES) + ROLLING_WINDOW
    train_history = train_df.iloc[:warmup].copy()
    train_eval = train_df.iloc[warmup:].copy()

    train_pred, train_actual = walk_forward(train_history, train_eval, FIXED_TUNED_PARAMS,
                                            feature_cols, target_col, stride=RETRAIN_STRIDE)
    train_metrics = calculate_metrics(train_actual, train_pred)

    # For the test evaluation, use train + validation as the starting training pool
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

    test_pred, test_actual = walk_forward(train_valid_df, test_df, FIXED_TUNED_PARAMS,
                                          feature_cols, target_col, stride=RETRAIN_STRIDE)
    test_metrics = calculate_metrics(test_actual, test_pred)

    return test_pred, test_actual, test_metrics, train_pred, train_actual, train_metrics












# Save forecasting results and evaluation outputs to CSV files           (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function exports:
# - validation performance of the fixed hyperparameters
# - final test predictions
# - actual target values
# - model configuration details
#
# The generated CSV files can later be used for analysis,
# visualisation, and comparison with other models.
# --- Save outputs ---

def save_outputs(full_df, train_len, valid_len, test_pred, test_actual, fixed_params, valid_metrics, target_col):
    # Save the validation results for the fixed hyperparameter config.
    # This replaces the grid search results CSV from the optimised version.
    valid_results = dict(fixed_params)
    valid_results["validation_rmse"] = valid_metrics["RMSE"]
    valid_results["validation_mae"] = valid_metrics["MAE"]
    valid_results["validation_mape"] = valid_metrics["MAPE"]
    valid_results["validation_directional_accuracy"] = valid_metrics["Directional_Accuracy"]

    valid_results_df = pd.DataFrame([valid_results])
    valid_results_df.to_csv("xgboost_fixed_params_results_1.csv", index=False)

    # Save the test predictions with the actual values and hyperparameter config
    start_idx = train_len + valid_len
    pred_df = full_df.iloc[start_idx:start_idx + len(test_pred)].copy()
    pred_df["actual_log_return_t_plus_1"] = test_actual
    pred_df["predicted_log_return_t_plus_1"] = test_pred

    # Add hyperparameter columns so we know what config produced these predictions
    for k, v in fixed_params.items():
        pred_df["best_" + k] = v  # column prefix kept as "best_" for compatibility

    pred_df.to_csv("xgboost_test_predictions_1.csv", index=False)

    logging.info("Saved -> xgboost_fixed_params_results_1.csv")
    logging.info("Saved -> xgboost_test_predictions_1.csv")













# Execute the complete XGBoost forecasting pipeline                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------
# This function controls the full forecasting workflow:
# - load and preprocess the data
# - generate model features
# - split the dataset
# - evaluate the baseline XGBoost model
# - compute forecasting metrics
# - print results
# - save outputs to CSV files
#
# It acts as the central controller of the entire forecasting process.
# --- Main ---

def xgboost_forecast():
    # Load and prepare the data
    df = load_data(CSV_PATH)
    feature_cols = get_feature_cols(df)
    target_col = "target_log_return_t_plus_" + str(FORECAST_HORIZON)

    # Split into train, validation, and test sets
    train_df, valid_df, test_df = split_data(df)

    # Evaluate the fixed hyperparameters on the validation set (no search done here)
    valid_pred, valid_actual, valid_metrics = evaluate_on_validation(
        train_df, valid_df, feature_cols, target_col
    )

    # Evaluate on the train and test sets
    test_pred, test_actual, test_metrics, train_pred, train_actual, train_metrics = fit_and_evaluate(
        train_df, valid_df, test_df, feature_cols, target_col
    )

    # Log all results
    logging.info("----- TRAIN RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for k, v in train_metrics.items():
        logging.info("%-25s: %.8f", k, v)

    logging.info("----- VALIDATION RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for k, v in valid_metrics.items():
        logging.info("%-25s: %.8f", k, v)

    logging.info("----- FINAL TEST RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for k, v in test_metrics.items():
        logging.info("%-25s: %.8f", k, v)

    # Print formatted summaries to the terminal
    print("\n" + "=" * 56)
    print("  TRAIN RESULTS  -  XGBoost (t+1 forecast)")
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
    print("  VALIDATION RESULTS  -  XGBoost (t+1 forecast)")
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
    print("  FINAL TEST RESULTS  -  XGBoost (t+1 forecast)")
    print("  [Before Hyperparameter Optimisation - Fixed Params]")
    print("=" * 56)
    print(f"  Fixed params            : {FIXED_TUNED_PARAMS}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 56)

    # Save results to CSV files
    save_outputs(
        full_df=df,
        train_len=len(train_df),
        valid_len=len(valid_df),
        test_pred=test_pred,
        test_actual=test_actual,
        fixed_params=FIXED_TUNED_PARAMS,
        valid_metrics=valid_metrics,
        target_col=target_col,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    xgboost_forecast()
