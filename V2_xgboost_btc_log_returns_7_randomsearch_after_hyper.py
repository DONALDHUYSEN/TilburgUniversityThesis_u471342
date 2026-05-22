# V2_xgboost_btc_log_returns_7_randomsearch_after_hyper.py
#
# XGBoost model to predict Bitcoin log returns at t+7.
#
# VERSION: V2 — After hyperparameter optimisation (random search)
#
# CHANGES IN V2 (relative to V1):
#   1. Processed datasets (train / validation / test) are saved to CSV after
#      feature engineering and the chronological 70/15/15 split.
#      File names embed the forecast horizon:
#        train_t7.csv  |  valid_t7.csv  |  test_t7.csv
#      These files can be loaded directly for SHAP analysis without
#      re-running feature engineering.
#
#   2. The trained final model (retrained on train+valid before the test walk)
#      is saved to disk as  xgb_model_t7.json  so it can be reloaded for
#      SHAP calculations without retraining.
#
#   3. No target leakage: all transformations happen before the split;
#      feature columns are identical across the three saved CSV files.
#
#   4. Output file names for predictions and search results now include the
#      horizon tag for consistency with V2 naming conventions.
#
# UNCHANGED from V1:
#   - t+7-specific settings: LAG_FEATURES = [1..7, 14, 21], ROLLING_WINDOW = 7
#   - Random search hyperparameter optimisation (60 unique combinations)
#   - Walk-forward validation on the validation set
#   - Rolling out-of-sample test evaluation
#   - Performance metrics: MAE, RMSE, MAPE, Directional Accuracy
#   - CSV output for predictions and search results
#
# Compared to the t+1 version, the following things differ (unchanged from V1):
#   1. FORECAST_HORIZON = 7  (target is log return 7 days ahead)
#   2. LAG_FEATURES = [1,2,3,4,5,6,7,14,21]  (extended look-back)
#   3. ROLLING_WINDOW = 7  (weekly volatility window)
#   4. Rolling feature column: rolling_vol_7  (vs rolling_vol_5 in t+1)
#
# Hyperparameter search space (unchanged):
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

FORECAST_HORIZON      = 7   # predict t+7 log return
RANDOM_SEARCH_STRIDE  = 10  # retrain every Nth step during random search (faster)
RETRAIN_STRIDE        = 1   # retrain at every step during final evaluation
EARLY_STOP_ROUNDS     = 20  # stop early if no improvement after this many rounds
EARLY_STOP_WINDOW     = 30  # how many recent rows to use as the early-stopping eval set

N_RANDOM_COMBINATIONS = 60  # number of random hyperparameter combinations to try

# Extended lag features for the t+7 forecast horizon:
#   Lags 1-7  : full week of history
#   Lag 14    : two weeks ago
#   Lag 21    : three weeks ago
LAG_FEATURES   = [1, 2, 3, 4, 5, 6, 7, 14, 21]

# 7-day rolling volatility window to match the weekly horizon
ROLLING_WINDOW = 7

RANDOM_SEED = 42

# Hyperparameter search space — unchanged from the t+1 version
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


# ==============================================================================
# HELPERS — horizon-aware naming                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This helper creates a short label for the current forecast horizon.
# In this script, FORECAST_HORIZON is 7, so the function returns "t7".
# The tag is used in filenames for saved datasets, predictions, search results,
# and the final trained XGBoost model.
#
# Using one central helper keeps the file naming consistent throughout the script.
# It also prevents outputs from different forecast horizons from overwriting each
# other when t+1, t+7, and t+30 experiments are run separately
def horizon_tag():
    """Return a short string like 't7' for the current horizon."""
    return f"t{FORECAST_HORIZON}"



# Create the target column name for the t+7 forecast         (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This helper builds the target column name based on the forecast horizon.
# In this script, it returns "target_log_return_t_plus_7".
#
# The same target name is reused during target creation, feature exclusion,
# model training, walk-forward evaluation, and output saving.
# Keeping this naming logic in one function reduces the risk of inconsistent
# column references elsewhere in the code.
def target_col_name():
    """Canonical name for the target column: 'target_log_return_t_plus_7'."""
    return f"target_log_return_t_plus_{FORECAST_HORIZON}"














# ==============================================================================
# STEP 1 — LOAD DATA                     (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function reads the raw Bitcoin dataset from the CSV file.
# It sorts the rows by date so the chronological order is preserved.
# The closing price column is checked, converted to numeric format,
# and rows with invalid prices are removed.
#
# Daily log returns are calculated from the closing price series.
# The t+7 target is created by shifting log returns seven rows backwards,
# meaning each current row is aligned with the return seven days ahead.
#
# Rows without a valid log return or future target are removed.
# The returned dataframe is ready for feature engineering.
def load_data(csv_path):
    """
    Read the CSV, calculate daily log returns, and create the target column.
    The target is the log return FORECAST_HORIZON (7) days in the future.

    shift(-7) moves the value 7 rows up so it aligns with the current row.
    This means: "what will the log return be 7 days from today?"
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
# STEP 2 — BUILD FEATURES                 (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function creates the predictive features used by the XGBoost model.
# All features are based only on past information to avoid target leakage.
# Text columns are converted into numeric category codes where needed.
#
# Lagged log returns are added for lags 1 to 7, plus lag 14 and lag 21.
# This gives the model information about the previous week and earlier patterns.
# A 7-day rolling volatility feature is also created and shifted by one day.
#
# Other numeric variables are lagged by one day so only prior values are used.
# Rows with NaN values caused by shifting are removed at the end.
def build_features(df):
    """
    Add new columns to the dataframe that the model will use as inputs.
    All features are lagged so we never accidentally use future data (data leakage).

    For t+7, we use more lag features than t+1:
      - Lags 1 to 7: covers the full past week
      - Lag 14: two weeks ago
      - Lag 21: three weeks ago
    This gives the model a richer view of recent history.

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
    # For t+7 we go up to lag_21 to capture more historical context.
    for lag in LAG_FEATURES:
        col_name = "lag_" + str(lag)
        df[col_name] = df["log_return"].shift(lag)

    # --- Rolling volatility ---
    # Standard deviation of the last 7 returns (shifted by 1 to avoid leakage).
    # Window is 7 here (vs 5 in the t+1 version) to match the weekly horizon.
    df["rolling_vol_7"] = df["log_return"].shift(1).rolling(window=ROLLING_WINDOW).std()

    # --- Lag all other numeric columns by 1 day ---
    already_created = (
        ["log_return", target_col_name(), "rolling_vol_7"]
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
# STEP 3 — IDENTIFY FEATURE COLUMNS              (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function identifies which columns should be used as model inputs.
# It excludes the date, closing price, raw log return, target variable,
# and asset identifier because these should not directly enter the model.
#
# Any column starting with "target_" is also excluded as an extra protection
# against accidental target leakage.
# The resulting feature list is used consistently for training, validation,
# testing, saved datasets, and later SHAP analysis.
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
# STEP 4 — SPLIT INTO TRAIN / VALIDATION / TEST          (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function divides the fully engineered dataframe into three chronological
# subsets using the 70/15/15 split.
# The training set is used for model fitting.
# The validation set is used during random search hyperparameter optimisation.
# The test set is kept aside for the final out-of-sample evaluation.
#
# No shuffling is applied because the data is time-dependent.
# This keeps the forecasting setup realistic by ensuring the model only learns
# from past observations when predicting future returns.
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
# STEP 4b — SAVE PROCESSED DATASETS  (NEW IN V2)         (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function saves the fully engineered and chronologically split datasets.
# The output files are named using the horizon tag, for example train_t7.csv.
#
# These files contain all feature columns, target values, and aligned rows.
# Saving them makes the workflow more reproducible and avoids having to rerun
# feature engineering for later analysis.
#
# The files are especially useful for SHAP analysis, since the test set can be
# loaded directly with the same feature structure used during training.
def save_datasets(train_df, valid_df, test_df):
    """
    Persist the three fully-engineered, correctly-split dataframes to CSV.

    File names embed the forecast horizon so that different horizons do not
    overwrite each other:
        train_t7.csv   valid_t7.csv   test_t7.csv

    These files can be loaded directly by a SHAP analysis script:
        test_df = pd.read_csv("test_t7.csv")
        X_test  = test_df[feature_cols]   # same feature_cols list used here
        shap_values = explainer.shap_values(X_test)

    No target leakage: all transformations were applied before the split, and
    the target column (shift(-7)) was created before any lagged features so it
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
# STEP 5 — METRICS                      (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function computes Mean Absolute Percentage Error for the forecasts.
# Since Bitcoin log returns can be very close to zero, direct division by the
# actual value can create unstable or extremely large percentage errors.
#
# To prevent this, the denominator is clipped using a small epsilon value.
# This avoids division by zero while keeping the metric interpretable.
#
# The returned value is the average percentage error across all predictions.
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







# Calculate directional accuracy              (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function checks whether the model predicts the correct return direction.
# It compares the sign of the predicted t+7 log return with the sign of the
# actual t+7 log return.
#
# A prediction is counted as correct when both signs match.
# The final score is returned as a percentage of correct directional forecasts.
#
# This metric is useful in financial forecasting because direction can be
# practically relevant even when exact return values are difficult to predict.
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







# Calculate model evaluation metrics              (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function computes the main metrics used to evaluate forecasting quality.
# MAE measures the average absolute error.
# RMSE gives extra weight to larger forecasting errors.
# MAPE gives a percentage-based error estimate using the safe MAPE helper.
# Directional Accuracy measures whether the predicted sign is correct.
#
# The metrics are returned in a dictionary so they can be logged, printed,
# saved to CSV files, and compared with other models or forecast horizons.
def calculate_metrics(y_true, y_pred):
    """Compute all four evaluation metrics and return them as a dictionary."""
    return {
        "MAE"                 : mean_absolute_error(y_true, y_pred),
        "RMSE"                : sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE"                : safe_mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }









# ==============================================================================
# STEP 6 — TRAIN A SINGLE XGBOOST MODEL                 (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function fits one XGBoost model using the supplied features, targets,
# and hyperparameter configuration.
#
# If enough observations are available, the final part of the training data
# is used as a temporary evaluation set for early stopping.
# Early stopping stops training if validation performance does not improve
# after a fixed number of rounds, reducing overfitting risk.
#
# If the dataset is too small for early stopping, the model is trained normally.
# The fitted model is returned for prediction.
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
# STEP 7 — WALK-FORWARD EVALUATION                  (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function simulates a realistic time-series forecasting process.
# It moves through the evaluation set one row at a time.
#
# At each step, the model is trained on all data available up to that moment.
# It then predicts the current unseen observation.
# After the prediction, the true row is added to the training pool.
#
# The stride controls how often the model is retrained. A stride of 1 retrains
# every step, while a larger stride speeds up validation during random search.
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
# STEP 8 — HYPERPARAMETER SEARCH (RANDOM SEARCH ON VALIDATION SET)          (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==============================================================================
# This function searches for the best XGBoost hyperparameter setup.
# It creates all possible combinations from the predefined parameter grid,
# then randomly samples 60 unique configurations using a fixed seed.
#
# Each sampled configuration is evaluated on the validation set through
# walk-forward forecasting.
# A faster retraining stride is used during random search to reduce runtime.
#
# Validation metrics are stored for every successful configuration.
# The configuration with the lowest validation RMSE is selected as best.
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









# ===============================================
# STEP 9 — EVALUATE ON TRAIN AND TEST SETS           (IMPROVED BY CLAUDE-Sonnet-4.6)
# =================================================
# This function evaluates the best hyperparameters found by random search.
# For the training evaluation, a warm-up period is used before predictions start.
# In this t+7 version, the warm-up accounts for the extended lag structure.
#
# For the test evaluation, train and validation data are combined first.
# The model then walks forward through the held-out test set using stride=1.
#
# After evaluation, a final model is trained on train+validation data and saved.
# The function returns predictions, actuals, and metrics for train and test.
def fit_and_evaluate(train_df, valid_df, test_df, best_params, feature_cols, target_col):
    """
    Evaluate the best hyperparameters on both the training set (in-sample)
    and the test set (out-of-sample).

    For training: we skip the first few rows as a warm-up window, then walk forward
    through the rest. This ensures the model always has enough history to start.
    warmup_size uses max(LAG_FEATURES) which is 21 for the t+7 version.

    For test: we combine train + validation as the starting training pool,
    then walk forward through the test set.

    Additionally (NEW IN V2): after completing the walk-forward test evaluation,
    we retrain a single final model on the full train+valid set and save it to
    disk as  xgb_model_t7.json.  This model can be reloaded for SHAP analysis
    without re-running the entire pipeline.
    """

    # --- Train evaluation ---
    # warmup_size accounts for the larger max lag (21) in this t+7 version
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
    #   model.load_model("xgb_model_t7.json")
    #   explainer = shap.TreeExplainer(model)
    save_final_model(train_and_valid, best_params, feature_cols, target_col)

    return (test_predictions, test_actuals, test_metrics,
            train_predictions, train_actuals, train_metrics)






# ====================================================================
# STEP 9b — SAVE FINAL TRAINED MODEL  (NEW IN V2)            (IMPROVED BY CLAUDE-Sonnet-4.6)
# =============================================================
# This function trains one final model on the combined train and validation set.
# It uses the best hyperparameters selected during random search.
#
# The saved model is written in XGBoost JSON format as xgb_model_t7.json.
# This allows the model to be reloaded later without repeating training.
#
# The saved model is mainly intended for post-hoc SHAP analysis.
# It is separate from the walk-forward models, which are repeatedly retrained.
def save_final_model(train_and_valid_df, best_params, feature_cols, target_col):
    """
    Train one final XGBoost model on the complete train+validation set using
    the best hyperparameters found during random search, then save it to disk.

    The saved file uses XGBoost's native JSON format, which is portable and
    version-stable.  Reload it with:
        model = xgb.XGBRegressor()
        model.load_model("xgb_model_t7.json")

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











# ==============================================================================
# STEP 10 — SAVE RESULTS TO CSV                       (IMPROVED BY CLAUDE-Sonnet-4.6)
# ======================================================================
# CSV files.
# First, it saves all random search configurations and their validation metrics.
# Second, it saves the final test predictions alongside the actual t+7 returns.
#
# The selected best hyperparameters are also added to the prediction file.
# Output file names include the t7 horizon tag to avoid confusion with other runs.
#
# These files can later be used for plotting, thesis tables, error analysis,
# and comparison with other models or horizons.
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
# MAIN                          (IMPROVED BY CLAUDE-Sonnet-4.6)
# ==========================================================
# Run the complete t+7 XGBoost after-optimisation pipeline
# ==============================================================================
# This function is the main controller of the full experiment.
# It loads the raw data, creates the t+7 target, builds features, and selects
# the final model input columns.
#
# It then splits the dataset chronologically and saves the processed datasets.
# Random search is performed on the validation set to find the best parameters.
#
# The best model is evaluated on train, validation, and test data.
# Finally, results are printed, logged, and exported to CSV files.
def xgboost_forecast():
    """
    Run the full pipeline from loading data to saving results.

    Order of operations (critical for correctness):
      1. Load raw data, compute log returns, create target with shift(-7)
      2. Build all lagged / rolling features  [no split yet — avoids leakage]
      3. Identify feature columns
      4. Split into train / valid / test chronologically (70 / 15 / 15)
      5. Save train_t7.csv, valid_t7.csv, test_t7.csv      [NEW IN V2]
      6. Random search on validation set to find best hyperparameters
      7. Walk-forward evaluation on train and test sets
      8. Save final model xgb_model_t7.json                [NEW IN V2]
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
    # fit_and_evaluate also saves the final model (xgb_model_t7.json)
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
# SHAP USAGE GUIDE  (run separately after this script completes)
# ==============================================================================
#
# The code below is NOT executed automatically.  Copy it into a new notebook
# or script to compute SHAP feature importance on the saved test set.
#
#   import shap
#   import xgboost as xgb
#   import pandas as pd
#
#   # 1. Reload the saved test set (already feature-engineered, no leakage)
#   test_df = pd.read_csv("test_t7.csv")
#
#   # 2. Reconstruct the feature column list (same logic as get_feature_cols)
#   EXCLUDE = {"Date", "Close", "log_return", "asset"}
#   feature_cols = [c for c in test_df.columns
#                   if c not in EXCLUDE and not c.startswith("target_")]
#   X_test = test_df[feature_cols]
#
#   # 3. Load the saved final model
#   model = xgb.XGBRegressor()
#   model.load_model("xgb_model_t7.json")
#
#   # 4. Compute SHAP values
#   explainer   = shap.TreeExplainer(model)
#   shap_values = explainer.shap_values(X_test)
#
#   # 5. Visualise
#   shap.summary_plot(shap_values, X_test)
#   shap.summary_plot(shap_values, X_test, plot_type="bar")
#
#   # To also run SHAP on the training set:
#   train_df = pd.read_csv("train_t7.csv")
#   X_train  = train_df[feature_cols]
#   shap_values_train = explainer.shap_values(X_train)
#   shap.summary_plot(shap_values_train, X_train)
