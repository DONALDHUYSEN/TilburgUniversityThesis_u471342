# V2_xgboost_btc_log_returns_1_randomsearch_after_hyper.py
#
# XGBoost model to predict Bitcoin log returns at t+N.
#
# VERSION: V2 — After hyperparameter optimisation (random search)
#
# CHANGES IN V2 (relative to V1):
#   1. Processed datasets (train / validation / test) are saved to CSV after
#      feature engineering and the chronological 70/15/15 split.
#      File names embed the forecast horizon, e.g.:
#        train_t1.csv  |  valid_t1.csv  |  test_t1.csv
#      This makes them directly reusable for SHAP analysis without
#      re-running feature engineering.
#
#   2. Full horizon-awareness: FORECAST_HORIZON controls shift(-N), target
#      column name, and all output file names, so the pipeline is identical
#      for t+1, t+7, t+30, etc.
#
#   3. The trained final model (retrained on train+valid before the test walk)
#      is saved to disk as  xgb_model_t{N}.json  so it can be reloaded for
#      SHAP calculations without retraining.
#
#   4. No target leakage: all transformations happen before the split;
#      feature columns are identical across the three saved CSV files.
#
# UNCHANGED from V1:
#   - Random search hyperparameter optimisation (60 unique combinations)
#   - Walk-forward validation on the validation set
#   - Rolling out-of-sample test evaluation
#   - Performance metrics: MAE, RMSE, MAPE, Directional Accuracy
#   - CSV output for predictions and search results
#
# Hyperparameter search space:
#   n_estimators     : [100, 200, 300, 500]
#   max_depth        : [3, 5, 7, 10]
#   learning_rate    : [0.01, 0.05, 0.1, 0.2]
#   subsample        : [0.6, 0.8, 1.0]
#   colsample_bytree : [0.6, 0.8, 1.0]

import logging
import random
import warnings
from itertools import product
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

FORECAST_HORIZON      = 1   # predict t+N log return  (change to 7 or 30 as needed)
RANDOM_SEARCH_STRIDE  = 10  # retrain every Nth step during random search (faster)
RETRAIN_STRIDE        = 1   # retrain at every step during final evaluation
EARLY_STOP_ROUNDS     = 20  # stop early if no improvement after this many rounds
EARLY_STOP_WINDOW     = 30  # how many recent rows to use as the early-stopping eval set

# We sample exactly 60 random combinations from the full grid.
N_RANDOM_COMBINATIONS = 60

LAG_FEATURES   = [1, 2, 3, 4, 5]   # lagged log returns: t-1 ... t-5
ROLLING_WINDOW = 5                  # rolling volatility window

RANDOM_SEED = 42

# Hyperparameter search space
PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 5, 7, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# XGBoost settings that never change
FIXED_PARAMS = {
    "objective"   : "reg:squarederror",
    "eval_metric" : "rmse",
    "tree_method" : "hist",
    "random_state": RANDOM_SEED,
    "verbosity"   : 0,
}








# ================================================ ==============================
# HELPERS — horizon-aware naming                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Generate a short horizon label for file naming and logging
# ==============================================================================
# This helper function creates a compact identifier based on the selected
# forecast horizon.
#
# Examples:
# - FORECAST_HORIZON = 1   -> "t1"
# - FORECAST_HORIZON = 7   -> "t7"
# - FORECAST_HORIZON = 30  -> "t30"
#
# The returned tag is reused throughout the pipeline for:
# - output CSV file names
# - saved model names
# - logging messages
#
# This ensures that outputs from different forecast horizons do not overwrite
# each other and remain easy to identify.
def horizon_tag():
    """Return a short string like 't1', 't7', or 't30' for the current horizon."""
    return f"t{FORECAST_HORIZON}"







# Create the canonical target column name                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This helper function generates the name of the prediction target column
# based on the selected forecast horizon.
#
# Examples:
# - t+1  -> "target_log_return_t_plus_1"
# - t+7  -> "target_log_return_t_plus_7"
#
# Using a dynamic naming system ensures that the same pipeline code works
# for multiple forecasting horizons without manual rewriting.
#
# The returned column name is reused consistently during:
# - target creation
# - feature selection
# - training
# - evaluation
# - CSV export
def target_col_name():
    """Canonical name for the target column, e.g. 'target_log_return_t_plus_1'."""
    return f"target_log_return_t_plus_{FORECAST_HORIZON}"











# ===================================
# STEP 1 — LOAD DATA                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# =================
# Load Bitcoin data and create forecasting targets
# ==============================================================================
# This function loads the raw Bitcoin dataset and prepares the initial
# forecasting target for the XGBoost model.
#
# Main processing steps:
# - load the CSV file
# - sort observations chronologically
# - clean invalid price values
# - calculate daily log returns
# - create the future prediction target using shift(-FORECAST_HORIZON)
#
# The target shifting aligns future log returns with current observations,
# allowing the model to learn how present information relates to future
# Bitcoin price movements.
#
# The function returns a cleaned dataframe ready for feature engineering.
def load_data(csv_path):
    """
    Read the CSV, calculate daily log returns, and create the target column.
    The target is the log return FORECAST_HORIZON days in the future.

    Using shift(-FORECAST_HORIZON) ensures the pipeline is correct for any
    horizon (t+1, t+7, t+30, …) without any manual edits beyond FORECAST_HORIZON.
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

    # The target is the log return FORECAST_HORIZON days from now.
    # shift(-FORECAST_HORIZON) moves the future value up to align with the current row.
    target_col = target_col_name()
    df[target_col] = df["log_return"].shift(-FORECAST_HORIZON)

    # Drop rows where we couldn't compute a return or a target
    df = df.dropna(subset=["log_return", target_col]).reset_index(drop=True)

    logging.info("Rows after loading: %d", len(df))
    return df











# ==============================================================================
# STEP 2 — BUILD FEATURES
# Build predictive input features for XGBoost                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function generates all model input features while preventing
# data leakage from future observations.
#
# The feature engineering process includes:
# - encoding categorical variables into numeric values
# - creating lagged log return features (t-1 to t-5)
# - calculating rolling volatility features
# - creating lagged versions of other numeric variables
#
# All engineered features are based only on historical information
# available before the prediction timestep.
#
# Rows containing NaN values created by shifting operations are removed
# before returning the final feature-enhanced dataframe.
def build_features(df):
    """
    Add new columns to the dataframe that the model will use as inputs.
    All features are lagged so we never accidentally use future data (data leakage).

    This function must be called BEFORE the train/valid/test split.
    The resulting dataframe is what gets split and saved to CSV.
    """
    df = df.copy()

    # XGBoost needs numbers. Convert any text columns (like 'bull'/'bear') to integers.
    text_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in text_columns:
        if col not in [DATE_COLUMN, "asset"]:
            df[col] = pd.Categorical(df[col]).codes

    # --- Lagged log returns ---
    # lag_1 = yesterday's return, lag_2 = two days ago, etc.
    for lag in LAG_FEATURES:
        col_name = "lag_" + str(lag)
        df[col_name] = df["log_return"].shift(lag)

    # --- Rolling volatility ---
    # Standard deviation of the last 5 returns (shifted by 1 to avoid leakage).
    df["rolling_vol_5"] = df["log_return"].shift(1).rolling(window=ROLLING_WINDOW).std()

    # --- Lag all other numeric columns by 1 day ---
    already_created = (
        ["log_return", target_col_name(), "rolling_vol_5"]
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
# STEP 3 — IDENTIFY FEATURE COLUMNS                       (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Identify the feature columns used by the model
# ==============================================================================
# This function returns a list of dataframe columns that should be used
# as predictive model inputs.
#
# Columns excluded from the feature set include:
# - dates
# - raw prices
# - target variables
# - raw log returns
# - identifier columns
#
# The resulting feature list is reused consistently throughout:
# - model training
# - evaluation
# - prediction generation
# - SHAP analysis
#
# This guarantees that all datasets use the exact same input structure.
# ==============================================================================
def get_feature_cols(df):
    """
    Return a list of column names that the model should use as inputs.
    We exclude the target, the date, the raw price, and the raw log return.

    This list is exactly what gets stored in the saved CSV files, ensuring that
    any SHAP analysis can simply call  df[feature_cols]  on the loaded CSV.
    """
    columns_to_exclude = [
        DATE_COLUMN,
        PRICE_COLUMN,
        "log_return",
        target_col_name(),
        "asset",
    ]

    feature_cols = []
    for col in df.columns:
        # Also skip any other target columns that might exist
        if col not in columns_to_exclude and not col.startswith("target_"):
            feature_cols.append(col)

    return feature_cols











# ==============================================================================
# STEP 4 — SPLIT INTO TRAIN / VALIDATION / TEST                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Split the dataset into train, validation, and test sets
# ==============================================================================
# This function divides the fully engineered dataset into three
# chronological subsets:
#
# - Train set       : used for model fitting
# - Validation set  : used during random hyperparameter search
# - Test set        : used for final out-of-sample evaluation
#
# No random shuffling is applied because time-series forecasting
# requires strict temporal ordering.
#
# The split is performed after feature engineering so that all datasets
# contain identical feature structures and aligned targets.
def split_data(df):
    """
    Split the data into three sequential chunks (no shuffling — time series!).
      - Train (70%): used to train the model
      - Validation (15%): used for the random hyperparameter search
      - Test (15%): final out-of-sample evaluation, only used once at the end

    The split is performed on the fully engineered dataframe, so every saved
    CSV already contains the complete feature set with no NaN rows.
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
# STEP 4b — SAVE PROCESSED DATASETS  (NEW IN V2)                 (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Save processed train, validation, and test datasets
# ==============================================================================
# This function exports the fully engineered datasets to CSV files
# after the chronological split has been completed.
#
# The saved files contain:
# - all engineered feature columns
# - target variables
# - aligned timestamps
#
# Forecast horizon tags are included in file names to prevent
# overwriting outputs from different experiments.
#
# These datasets are especially useful for:
# - SHAP analysis
# - reproducibility
# - debugging
# - downstream visualisation scripts
def save_datasets(train_df, valid_df, test_df):
    """
    Persist the three fully-engineered, correctly-split dataframes to CSV.

    File names embed the forecast horizon so that different horizons do not
    overwrite each other, e.g.:
        train_t1.csv   valid_t1.csv   test_t1.csv
        train_t7.csv   valid_t7.csv   test_t7.csv

    These files can be loaded directly by a SHAP analysis script:
        test_df = pd.read_csv("test_t1.csv")
        X_test  = test_df[feature_cols]   # same feature_cols list used here
        shap_values = explainer.shap_values(X_test)

    No target leakage: all transformations were applied before the split, and
    the target column (shift(-N)) was created before any lagged features so it
    is correctly aligned throughout.
    """
    tag = horizon_tag()

    train_path = f"train_{tag}.csv"
    valid_path = f"valid_{tag}.csv"
    test_path  = f"test_{tag}.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path,  index=False)

    logging.info("Saved processed datasets  ->  %s | %s | %s",
                 train_path, valid_path, test_path)











# ==============================================================================
# STEP 5 — METRICS                            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Compute MAPE while preventing division instability
# ==============================================================================
# Bitcoin log returns are often extremely small and may approach zero,
# which can cause instability in percentage-based error calculations.
#
# This function therefore applies a small epsilon threshold to the
# denominator before computing Mean Absolute Percentage Error (MAPE).
#
# The resulting metric provides a safer and more numerically stable
# estimate of percentage forecasting error.
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








# Measure directional forecasting accuracy            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function calculates how often the predicted Bitcoin return
# direction matches the true market direction.
#
# A prediction is counted as correct when:
# - both predicted and actual returns are positive
# - or both are negative
#
# Directional Accuracy is especially useful in financial forecasting
# because correctly predicting market direction may still have practical
# value even when numerical prediction errors remain large.
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








# Compute forecasting evaluation metrics                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function calculates the main statistical metrics used to evaluate
# XGBoost forecasting performance.
#
# The computed metrics include:
# - MAE
# - RMSE
# - MAPE
# - Directional Accuracy
#
# The results are returned as a dictionary so they can easily be:
# - logged
# - printed
# - compared
# - exported to CSV files
# =============================
def calculate_metrics(y_true, y_pred):
    """Compute all four evaluation metrics and return them as a dictionary."""
    return {
        "MAE"                 : mean_absolute_error(y_true, y_pred),
        "RMSE"                : sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE"                : safe_mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }










# ==============================================================================
# STEP 6 — TRAIN A SINGLE XGBOOST MODEL                   (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Train a single XGBoost forecasting model
# ==============================================================================
# This function fits one XGBoost regression model using the supplied
# training data and hyperparameter configuration.
#
# Early stopping is used to reduce overfitting:
# - the most recent observations are temporarily used as an evaluation set
# - training stops automatically if performance no longer improves
#
# If insufficient training data is available, the model is trained
# normally without early stopping.
#
# The trained XGBoost model is returned for forecasting use.
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
# STEP 7 — WALK-FORWARD EVALUATION                     (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Perform rolling walk-forward forecasting
# ==============================================================================
# This function simulates realistic forecasting conditions by stepping
# sequentially through unseen evaluation data.
#
# At each timestep:
# - the model is trained on all observations seen so far
# - a prediction is generated for the next unseen row
# - the true observation is added to the training pool
#
# The stride parameter controls how often retraining occurs:
# - stride=1  -> retrain every step (most accurate)
# - larger stride -> faster evaluation with less retraining
#
# This creates a rolling out-of-sample forecasting framework.
def walk_forward(train_df, eval_df, params, feature_cols, target_col, stride=1):
    """
    Simulate how the model would be used in real life.

    We step through the evaluation set one row at a time:
      1. Train the model on everything seen so far.
      2. Predict the next row.
      3. Add that row to the training pool.
      4. Repeat.

    stride=1 means we retrain at every single step.
    A larger stride (e.g. 10) retrains less often — faster but slightly less accurate.
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
# STEP 8 — HYPERPARAMETER SEARCH (RANDOM SEARCH ON VALIDATION SET)        (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Perform random search hyperparameter optimisation          
# ==============================================================================
# This function searches for the best XGBoost hyperparameter configuration
# using random search on the validation dataset.
#
# Main workflow:
# - generate all possible parameter combinations
# - randomly sample 60 unique combinations
# - evaluate each configuration using walk-forward forecasting
# - compute validation metrics
# - rank configurations using validation RMSE
#
# The configuration with the lowest validation RMSE is selected
# as the optimal XGBoost model.
#
# All search results are stored for later analysis and CSV export.
def random_search(train_df, valid_df, feature_cols, target_col):
    """
    Randomly sample N_RANDOM_COMBINATIONS unique hyperparameter combinations
    from the full search space and evaluate each one on the validation set.
    Returns the best combination and a dataframe with all results.
    """

    # Set the random seed so results are reproducible
    random.seed(RANDOM_SEED)

    # Step 1: Build the full list of every possible combination in the grid.
    all_combinations = list(product(*PARAM_GRID.values()))

    total_possible = len(all_combinations)
    logging.info("Total possible combinations in the grid: %d", total_possible)

    # Step 2: Figure out how many combinations we can actually sample.
    n_to_sample = min(N_RANDOM_COMBINATIONS, total_possible)

    # Step 3: Randomly sample n_to_sample unique combinations (no duplicates).
    sampled_combinations = random.sample(all_combinations, n_to_sample)

    param_names = list(PARAM_GRID.keys())

    logging.info(
        "Random search: evaluating %d unique combinations out of %d possible (stride=%d)",
        n_to_sample, total_possible, RANDOM_SEARCH_STRIDE,
    )

    # Step 4: Evaluate each sampled combination on the validation set.
    results = []

    for i in range(len(sampled_combinations)):
        combo = sampled_combinations[i]

        params = {}
        for j in range(len(param_names)):
            params[param_names[j]] = combo[j]

        logging.info("Trying combination %d / %d: %s", i + 1, n_to_sample, params)

        try:
            pred, actual = walk_forward(
                train_df, valid_df,
                params,
                feature_cols, target_col,
                stride=RANDOM_SEARCH_STRIDE,
            )

            metrics = calculate_metrics(actual, pred)

            row = {}
            row["n_estimators"]                    = params["n_estimators"]
            row["max_depth"]                       = params["max_depth"]
            row["learning_rate"]                   = params["learning_rate"]
            row["subsample"]                       = params["subsample"]
            row["colsample_bytree"]                = params["colsample_bytree"]
            row["validation_rmse"]                 = metrics["RMSE"]
            row["validation_mae"]                  = metrics["MAE"]
            row["validation_mape"]                 = metrics["MAPE"]
            row["validation_directional_accuracy"] = metrics["Directional_Accuracy"]

            results.append(row)

            logging.info(
                "  -> valid RMSE=%.8f | MAE=%.8f | MAPE=%.4f | DA=%.2f%%",
                metrics["RMSE"], metrics["MAE"],
                metrics["MAPE"], metrics["Directional_Accuracy"],
            )

        except Exception as exc:
            logging.warning("Combination %s failed with error: %s", params, exc)

    if not results:
        raise RuntimeError("No hyperparameter combination completed successfully.")

    # Sort all results by validation RMSE (lowest = best) and pick the top one.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("validation_rmse")
    results_df = results_df.reset_index(drop=True)

    best_params = {}
    best_params["n_estimators"]     = int(results_df.iloc[0]["n_estimators"])
    best_params["max_depth"]        = int(results_df.iloc[0]["max_depth"])
    best_params["learning_rate"]    = results_df.iloc[0]["learning_rate"]
    best_params["subsample"]        = results_df.iloc[0]["subsample"]
    best_params["colsample_bytree"] = results_df.iloc[0]["colsample_bytree"]

    logging.info(
        "Best params found by random search: %s  |  validation RMSE: %.8f",
        best_params, results_df.iloc[0]["validation_rmse"]
    )

    return best_params, results_df













# ==============================================================================
# STEP 9 — EVALUATE ON TRAIN AND TEST SETS                    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Evaluate the optimised XGBoost model on train and test datasets
# ==============================================================================
# This function evaluates the best hyperparameter configuration using
# rolling walk-forward forecasting.
#
# Two evaluations are performed:
# - in-sample evaluation on the training set
# - out-of-sample evaluation on the test set
#
# After the evaluation process:
# - a final XGBoost model is retrained on train+validation data
# - the trained model is saved to disk for later SHAP analysis
#
# The function returns predictions, actual values, and evaluation metrics
# for both datasets.
def fit_and_evaluate(train_df, valid_df, test_df, best_params, feature_cols, target_col):
    """
    Evaluate the best hyperparameters on both the training set (in-sample)
    and the test set (out-of-sample).

    For training: we skip the first few rows as a warm-up window, then walk forward
    through the rest. This ensures the model always has enough history to start.

    For test: we combine train + validation as the starting training pool,
    then walk forward through the test set.

    Additionally (NEW IN V2): after completing the walk-forward test evaluation,
    we retrain a single final model on the full train+valid set and save it to
    disk as  xgb_model_t{N}.json.  This model can be reloaded for SHAP analysis
    without re-running the entire pipeline.
    """

    # --- Train evaluation ---
    warmup_size  = EARLY_STOP_WINDOW + max(LAG_FEATURES) + ROLLING_WINDOW
    train_warmup = train_df.iloc[:warmup_size].copy()
    train_eval   = train_df.iloc[warmup_size:].copy()

    train_predictions, train_actuals = walk_forward(
        train_warmup, train_eval,
        best_params,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )
    train_metrics = calculate_metrics(train_actuals, train_predictions)

    # --- Test evaluation ---
    train_and_valid = pd.concat([train_df, valid_df], ignore_index=True)

    test_predictions, test_actuals = walk_forward(
        train_and_valid, test_df,
        best_params,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )
    test_metrics = calculate_metrics(test_actuals, test_predictions)

    # --- NEW IN V2: save the final model trained on train+valid ---
    # This model is reusable for SHAP analysis — load it with:
    #   import xgboost as xgb
    #   model = xgb.XGBRegressor()
    #   model.load_model("xgb_model_t1.json")
    #   explainer = shap.TreeExplainer(model)
    save_final_model(train_and_valid, best_params, feature_cols, target_col)

    return (test_predictions, test_actuals, test_metrics,
            train_predictions, train_actuals, train_metrics)











# ==============================================================================
# STEP 9b — SAVE FINAL TRAINED MODEL  (NEW IN V2)                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Train and save the final XGBoost model
# ==============================================================================
# This function trains one final XGBoost model using the complete
# train+validation dataset and the best hyperparameters found during
# random search.
#
# The resulting model is saved as a JSON file so it can later be:
# - reloaded without retraining
# - used for SHAP analysis
# - reused in external scripts
#
# This saved model is separate from the walk-forward models that are
# repeatedly retrained during forecasting evaluation.
def save_final_model(train_and_valid_df, best_params, feature_cols, target_col):
    """
    Train one final XGBoost model on the complete train+validation set using
    the best hyperparameters found during random search, then save it to disk.

    The saved file uses XGBoost's native JSON format, which is portable and
    version-stable.  Reload it with:
        model = xgb.XGBRegressor()
        model.load_model("xgb_model_t1.json")

    This model is meant for post-hoc SHAP analysis on the test set.
    It is NOT the walk-forward model (which is retrained at every step);
    it is the single best model that saw the most training data.
    """
    X_all = train_and_valid_df[feature_cols].values
    y_all = train_and_valid_df[target_col].values

    final_model = train_xgb(X_all, y_all, best_params)

    model_path = f"xgb_model_{horizon_tag()}.json"
    final_model.save_model(model_path)
    logging.info("Saved final model  ->  %s", model_path)















# =========================================== ==================================
# STEP 10 — SAVE RESULTS TO CSV                       (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# Save random search results and test predictions
# ==============================================================================
# This function exports:
# - all random search validation results
# - final test predictions
# - actual target values
# - selected hyperparameters
#
# Forecast horizon tags are included in file names so outputs from
# different experiments remain separated.
#
# The generated CSV files can later be used for:
# - result analysis
# - plotting
# - comparison with other models
# - thesis reporting
def save_outputs(full_df, train_len, valid_len,
                 test_pred, test_actual,
                 best_params, search_results, target_col):
    """
    Save two CSV files:
      1. All random search results (all 60 combinations tried + their validation metrics).
      2. The test set predictions alongside the actual values.

    File names embed the horizon tag so t+1 and t+7 runs do not collide.
    """
    tag = horizon_tag()

    # --- File 1: random search results ---
    search_path = f"xgboost_random_search_results_{tag}.csv"
    search_results.to_csv(search_path, index=False)

    # --- File 2: test predictions ---
    start_idx = train_len + valid_len
    pred_df   = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()

    pred_df["actual_log_return"]    = test_actual
    pred_df["predicted_log_return"] = test_pred

    for param_name, param_value in best_params.items():
        pred_df["best_" + param_name] = param_value

    pred_path = f"xgboost_test_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    logging.info("Saved  ->  %s", search_path)
    logging.info("Saved  ->  %s", pred_path)









# ==============================================================================
# MAIN                     
# ==============================================================================
# Execute the complete XGBoost forecasting pipeline            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function controls the entire forecasting workflow from start to finish.
#
# Main pipeline steps:
# - load and preprocess Bitcoin data
# - generate lagged and rolling features
# - split datasets chronologically
# - save processed datasets
# - perform random search optimisation
# - evaluate forecasting performance
# - save the final trained model
# - print and log all evaluation metrics
# - export predictions and search results
#
# It acts as the central controller of the full XGBoost experiment.
def xgboost_forecast():
    """
    Run the full pipeline from loading data to saving results.

    Order of operations (critical for correctness):
      1. Load raw data, compute log returns, create target with shift(-N)
      2. Build all lagged / rolling features  [no split yet — avoids leakage]
      3. Identify feature columns
      4. Split into train / valid / test chronologically (70 / 15 / 15)
      5. Save train_{tag}.csv, valid_{tag}.csv, test_{tag}.csv  [NEW IN V2]
      6. Random search on validation set to find best hyperparameters
      7. Walk-forward evaluation on train and test sets
      8. Save final model xgb_model_{tag}.json                  [NEW IN V2]
      9. Re-evaluate validation set at stride=1 for the summary print
     10. Log and print all metrics
     11. Save prediction CSVs and random search results
    """

    target_col = target_col_name()
    tag        = horizon_tag()

    # --- Step 1-2: Load and engineer features ---
    df = load_data(CSV_PATH)
    df = build_features(df)
    feature_cols = get_feature_cols(df)

    logging.info("Forecast horizon        : t+%d  (%s)", FORECAST_HORIZON, tag)
    logging.info("Number of feature cols  : %d",          len(feature_cols))
    logging.info("Feature columns         : %s",          feature_cols)

    # --- Step 3: Split (time-ordered, no shuffle) ---
    train_df, valid_df, test_df = split_data(df)

    # --- Step 4 (NEW): Persist the split datasets ---
    # These files are the ground truth for any downstream SHAP analysis.
    # They contain every feature column used by the model, plus the target,
    # and are free of any post-split transformations.
    save_datasets(train_df, valid_df, test_df)

    # --- Step 5: Random search on validation set ---
    best_params, search_results = random_search(
        train_df, valid_df, feature_cols, target_col
    )

    # --- Step 6: Walk-forward evaluation on train and test sets ---
    # fit_and_evaluate also saves the final model (xgb_model_{tag}.json)
    (test_pred, test_actual, test_metrics,
     train_pred, train_actual, train_metrics) = fit_and_evaluate(
        train_df, valid_df, test_df, best_params, feature_cols, target_col
    )

    # --- Step 7: Re-evaluate validation set at stride=1 for the summary ---
    valid_pred, valid_actual = walk_forward(
        train_df, valid_df,
        best_params,
        feature_cols, target_col,
        stride=RETRAIN_STRIDE
    )
    valid_metrics = calculate_metrics(valid_actual, valid_pred)

    # --- Log all results ---
    logging.info("----- TRAIN RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for metric_name, metric_value in train_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info("----- VALIDATION RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for metric_name, metric_value in valid_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    logging.info("----- FINAL TEST RESULTS (walk-forward, t+%d) -----", FORECAST_HORIZON)
    for metric_name, metric_value in test_metrics.items():
        logging.info("%-25s: %.8f", metric_name, metric_value)

    # --- Print formatted summaries to the terminal ---
    horizon_label = f"t+{FORECAST_HORIZON} forecast"

    print("\n" + "=" * 60)
    print(f"  TRAIN RESULTS  -  XGBoost ({horizon_label})")
    print("  [After Hyperparameter Optimisation - Random Search]")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {train_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"  VALIDATION RESULTS  -  XGBoost ({horizon_label})")
    print("  [After Hyperparameter Optimisation - Random Search]")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {valid_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {valid_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {valid_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {valid_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print(f"  FINAL TEST RESULTS  -  XGBoost ({horizon_label})")
    print("  [After Hyperparameter Optimisation - Random Search]")
    print("=" * 60)
    print(f"  Best params             : {best_params}")
    print(f"  Forecast horizon        : t+{FORECAST_HORIZON}")
    print(f"  MAE                     : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                    : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                    : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy    : {test_metrics['Directional_Accuracy']:.2f}%")
    print("=" * 60)

    print(f"\n  Saved datasets     :  train_{tag}.csv | valid_{tag}.csv | test_{tag}.csv")
    print(f"  Saved model        :  xgb_model_{tag}.json")
    print(f"  Saved predictions  :  xgboost_test_predictions_{tag}.csv")
    print(f"  Saved search log   :  xgboost_random_search_results_{tag}.csv\n")

    # --- Save CSV outputs ---
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
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    xgboost_forecast()


# ==============================================================================
# SHAP USAGE GUIDE  (run separately after this script completes)   (THIS ONE IS NOT USED)
# ==============================================================================
#
# The code below is NOT executed automatically.  Copy it into a new notebook
# or script to compute SHAP feature importance on the saved test set.
#
#   import shap
#   import xgboost as xgb
#   import pandas as pd
#
#   FORECAST_HORIZON = 1   # must match the value used during training
#   tag = f"t{FORECAST_HORIZON}"
#
#   # 1. Reload the saved test set (already feature-engineered, no leakage)
#   test_df = pd.read_csv(f"test_{tag}.csv")
#
#   # 2. Reconstruct the feature column list (same logic as get_feature_cols)
#   EXCLUDE = {"Date", "Close", "log_return", "asset"}
#   feature_cols = [c for c in test_df.columns
#                   if c not in EXCLUDE and not c.startswith("target_")]
#   X_test = test_df[feature_cols]
#
#   # 3. Load the saved final model
#   model = xgb.XGBRegressor()
#   model.load_model(f"xgb_model_{tag}.json")
#
#   # 4. Compute SHAP values
#   explainer   = shap.TreeExplainer(model)
#   shap_values = explainer.shap_values(X_test)
#
#   # 5. Visualise
#   shap.summary_plot(shap_values, X_test)
#   shap.summary_plot(shap_values, X_test, plot_type="bar")
