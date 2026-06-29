"""
arima_btc_log_returns.py
=======================================
ARIMA model for Bitcoin daily log-return forecasting (t+1 prediction).

100-day fixed lookback window — at each evaluation position the
model is fitted on exactly the 100 most-recent observations that
precede that point, then a single 1-step-ahead forecast is made
for t+1. No rolling history accumulation occurs. This mirrors the
sliding-window (lookback=100) approach used in the updated LSTM
pipeline, ensuring methodological consistency across all models.

A fixed ARIMA(1, 0, 1) order is used throughout. Log returns are
typically stationary (d=0), and p=1, q=1 captures short-term
autoregressive and moving-average dynamics, making ARIMA(1,0,1)
a standard, well-motivated baseline for financial return series.
"""

import logging
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)



# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "btc_clean.csv"
PRICE_COLUMN = "Close"
DATE_COLUMN = "Date"

# Train / validation / test proportions (unchanged from original)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

# Fixed ARIMA order — used for all evaluation sets
# ARIMA(1, 0, 1): log returns are stationary (d=0); p=1 and q=1 capture
# short-term AR and MA dynamics. Standard baseline for financial return series.
ARIMA_ORDER = (1, 0, 1)

# fixed lookback window size
# ---------------------------------------------------------------------------
LOOKBACK = 100




# Data loading & preprocessing    
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------
# This function:
# 1. Reads the CSV file containing historical Bitcoin prices.
# 2. Converts the date column to datetime format and sorts the dataset
#    chronologically to preserve temporal order.
# 3. Checks whether the required closing price column exists.
# 4. Keeps only the relevant columns (Date and Close price).
# 5. Converts price values to numeric format and removes invalid rows.
# 6. Computes daily log returns using:
#       r_t = log(P_t / P_{t-1})
#    which transforms raw prices into stationary percentage-like changes.
# 7. Removes the first NaN row created by the differencing operation.
#
# The returned dataframe is therefore cleaned, chronologically ordered,
# and ready for time-series forecasting with ARIMA.

def load_data(csv_path):
    """Load CSV, sort by date, and compute daily log returns from Close price."""
    logging.info("Reading CSV file: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Parse and sort by date if the date column is present
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Column '{PRICE_COLUMN}' not found in {csv_path}")

    # Keep only the columns we need
    if DATE_COLUMN in df.columns:
        keep_cols = [DATE_COLUMN, PRICE_COLUMN]
    else:
        keep_cols = [PRICE_COLUMN]

    df = df[keep_cols].copy()
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
    df = df.dropna(subset=[PRICE_COLUMN]).reset_index(drop=True)

    # Compute daily log return: r_t = log(P_t / P_{t-1})
    df["log_return"] = np.log(df[PRICE_COLUMN]).diff()
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    logging.info("Loaded %d rows after computing log returns.", len(df))
    return df






# Train / validation / test split
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------
# This function divides the full time-series dataset into three sequential
# subsets while preserving chronological order:
#
# - Training set   : used to fit the ARIMA models
# - Validation set : used to report intermediate performance metrics
# - Test set       : used only for the final out-of-sample evaluation
#
# The split ratios are:
#   70% training
#   15% validation
#   15% testing
#
# Unlike random splitting, no shuffling is applied because time-series models
# must only learn from past observations when predicting future values.
#
# An additional safety check verifies that the training set contains at least
# LOOKBACK observations. This is necessary because the 100-day sliding-window
# forecasting approach requires a full historical context window before the
# first validation prediction can be generated.
def split_series(df):
    """
    Split the full log-return series into train, validation, and test subsets.

    The split proportions are unchanged from the original (70/15/15).
    A minor check ensures the training set is large enough to provide a full
    100-day context window for the very first evaluation step.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test  = df.iloc[valid_end:].copy()

    logging.info("Train size : %d", len(train))
    logging.info("Valid size : %d", len(valid))
    logging.info("Test size  : %d", len(test))

    # We need at least LOOKBACK rows before the first validation step
    if train_end < LOOKBACK:
        raise ValueError(
            f"Training set ({train_end} rows) is smaller than LOOKBACK ({LOOKBACK}). "
            "Reduce LOOKBACK or supply more data."
        )

    return train, valid, test







# Metric helpers              
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------
# These helper functions calculate the performance metrics used to evaluate
# the forecasting accuracy of the ARIMA model.
#
# The metrics serve different purposes:
#
# - MAE (Mean Absolute Error):
#     Measures the average absolute prediction error.
#     Easier to interpret because errors remain in the original log-return scale.
#
# - RMSE (Root Mean Squared Error):
#     Similar to MAE, but penalises larger prediction errors more heavily.
#     Particularly relevant for Bitcoin due to sudden volatile price movements.
#
# - MAPE (Mean Absolute Percentage Error):
#     Measures prediction error in percentage terms.
#     A small epsilon value is added in the denominator to avoid instability
#     when actual log returns are close to zero.
#
# - Directional Accuracy:
#     Measures how often the model correctly predicts the direction
#     of the market movement (positive or negative return).
#
# - R² (Coefficient of Determination):
#     Indicates how much variance in the true values is explained by the model.
#
# - Explained Variance:
#     Measures how well the model captures variation in the target series,
#     while being less sensitive to systematic prediction bias.
#
# Together, these metrics provide both statistical and practical insight
# into forecasting performance.
def safe_mape(y_true, y_pred, epsilon=1e-8):
    """MAPE with a small epsilon guard to avoid division by near-zero values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def directional_accuracy(y_true, y_pred):
    """Fraction of forecasts where the predicted sign matches the actual sign."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0


def calculate_metrics(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, Directional Accuracy, R2, and Explained Variance."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    da   = directional_accuracy(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    ev   = explained_variance_score(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": da,
        "R2": r2,
        "Explained_Variance": ev,
    }








# 100-day lookback forecasting 
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.
# ---------------------------------------------------------------------------
# This function performs 1-step-ahead forecasting using a rolling
# fixed-length historical window rather than an expanding walk-forward history.
#
# Forecasting procedure:
#
# 1. A window containing exactly `lookback` historical observations
#    is selected directly before the prediction point.
#
# 2. A new ARIMA(p,d,q) model is fitted only on this fixed window.
#
# 3. The model generates a single forecast for the next timestep (t+1).
#
# 4. The window then shifts forward by one observation and the process
#    repeats until the full evaluation series has been predicted.
#
# Unlike traditional walk-forward validation, the historical training size
# does not continuously expand over time. This ensures that every prediction
# is generated using the same amount of historical information, which improves
# methodological consistency across evaluation steps.
#
# The approach was specifically chosen to mirror the 100-day sliding-window
# methodology used in the LSTM and CNN-LSTM models. This makes comparisons
# between ARIMA and the deep learning architectures more consistent and fair.
#
# Additional safeguards:
# - Invalid model fits are caught with try/except blocks.
# - Failed forecasts are stored as NaN and removed afterwards.
# - Warnings from statsmodels are suppressed to reduce unnecessary output.
#
# Returns:
# - predictions : array containing the predicted log returns
# - actuals     : array containing the corresponding true log returns
def lookback_forecast(context_series, eval_series, order, lookback=LOOKBACK):
    """
    Fixed-window 1-step-ahead ARIMA forecasting.

    100-day lookback approach
    ------------------------------
    For each position i in eval_series:
        1. Build a window of exactly `lookback` observations ending just before i.
           The first `lookback` rows come from context_series. As i advances
           further into eval_series, the window slides forward and includes more
           rows from eval_series (true values), always keeping exactly 100 obs.
        2. Fit a fresh ARIMA(order) model on that fixed-length window.
        3. Produce a single 1-step-ahead forecast (the t+1 log return).
        4. Discard the fitted model — no state is carried forward.

    This is equivalent to the sliding-window (lookback=100) batch approach used
    in the LSTM pipeline, making evaluation methodologically consistent.

    Parameters
    ----------
    context_series : pd.Series
        Historical observations available before the eval set starts.
        Must contain at least `lookback` rows.
    eval_series : pd.Series
        The set of observations to predict (validation or test split).
    order : tuple
        ARIMA (p, d, q) order.
    lookback : int
        Fixed window size (default: 100).

    Returns
    -------
    predictions : np.ndarray  -- 1-step-ahead forecasts aligned to eval_series
    actuals     : np.ndarray  -- corresponding true log-return values
    """

    # Concatenate context + eval into one continuous array for easy slicing
    full_series = pd.concat([context_series, eval_series], axis=0).reset_index(drop=True)
    context_len = len(context_series)

    predictions = []

    for step in range(len(eval_series)):
        # Index of the observation we are trying to predict
        target_idx = context_len + step

        # Fixed lookback window: exactly the 100 rows immediately preceding target_idx
        window_start = target_idx - lookback
        window_end   = target_idx          # slice is exclusive at the end
        window = full_series.iloc[window_start:window_end].values

        if len(window) < lookback:
            # Safety guard — should not occur if the sanity check in split_series passed
            logging.warning(
                "Window at step %d has only %d rows (expected %d). Skipping.",
                step, len(window), lookback,
            )
            predictions.append(np.nan)
            continue

        # Fit ARIMA on the fixed 100-day window and forecast 1 step ahead (t+1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model  = ARIMA(window, order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=1)
                # Handle both Series and ndarray outputs from statsmodels
                if hasattr(forecast, "iloc"):
                    pred_value = float(forecast.iloc[0])
                else:
                    pred_value = float(forecast[0])
            except Exception as exc:
                logging.warning("ARIMA%s failed at step %d: %s", order, step, exc)
                pred_value = np.nan

        predictions.append(pred_value)

    predictions = np.asarray(predictions, dtype=float)
    actuals     = np.asarray(eval_series, dtype=float)

    # Drop positions where the model failed to produce a valid prediction
    valid_mask = ~np.isnan(predictions)
    return predictions[valid_mask], actuals[valid_mask]










# Evaluation on the train set (in-sample, 100-day lookback)       
# ---------------------------------------------------------------------------
# This function measures the in-sample performance of the fixed ARIMA(p,d,q)
# configuration using the same fixed 100-day sliding-window methodology
# applied during validation and testing.
#
# Evaluation procedure:
#
# 1. The first LOOKBACK observations are used as the initial historical
#    context window.
#
# 2. Starting from observation LOOKBACK + 1, the model repeatedly:
#       - fits an ARIMA model on the previous 100 observations
#       - predicts the next timestep (t+1 log return)
#
# 3. Predicted values are compared against the true log returns.
#
# 4. Forecasting metrics such as MAE, RMSE, MAPE, Directional Accuracy,
#    R², and Explained Variance are calculated.
#
# Using the same sliding-window structure across train, validation,
# and test evaluation ensures methodological consistency throughout
# the entire forecasting pipeline.
#
# Returns:
# - train_pred    : predicted log returns for the training set
# - train_actual  : true log returns corresponding to the predictions
# - train_metrics : dictionary containing all evaluation metrics
def evaluate_on_train(train_series, order):
    """
    Evaluate the fixed ARIMA order on the training set using the same
    100-day lookback approach.

    The first LOOKBACK rows are used as the initial context window.
    Predictions start from index LOOKBACK onward (train_series[LOOKBACK:]).
    This mirrors the sliding-window methodology used for validation/test.
    """
    # Context: the first LOOKBACK rows of train
    # Eval   : everything after that in the train set
    context = train_series.iloc[:LOOKBACK].reset_index(drop=True)
    eval_   = train_series.iloc[LOOKBACK:].reset_index(drop=True)

    train_pred, train_actual = lookback_forecast(context, eval_, order)

    train_metrics = calculate_metrics(train_actual, train_pred)
    return train_pred, train_actual, train_metrics









# Evaluation on the validation set with the fixed order     
# ---------------------------------------------------------------------------
# This function evaluates the fixed ARIMA(p,d,q) configuration on the
# validation dataset using the same fixed 100-day sliding-window
# forecasting methodology applied throughout the pipeline.
#
# Purpose of the validation evaluation:
# - Assess how well the model generalises to unseen data.
# - Provide a consistent comparison point between training and test results.
# - Generate validation metrics that can later be reported and analysed.
#
# Forecasting procedure:
# 1. Use the training series as historical context.
# 2. Apply the fixed 100-day lookback forecasting process on the
#    validation observations.
# 3. Compare predicted and actual log returns.
# 4. Compute evaluation metrics including MAE, RMSE, MAPE,
#    Directional Accuracy, R², and Explained Variance.
#
# Using the exact same evaluation structure across train, validation,
# and test datasets ensures methodological consistency and reduces the
# risk of unfair performance comparisons between datasets.
#
# Returns:
# - valid_pred    : predicted log returns for the validation set
# - valid_actual  : true log returns corresponding to the predictions
# - valid_metrics : dictionary containing all evaluation metrics
def evaluate_on_validation(train_series, valid_series, order):
    """
    Run the 100-day lookback forecast on the validation set using the
    fixed ARIMA order. This produces the validation metrics in the same
    format as the train and test result blocks.
    """
    valid_pred, valid_actual = lookback_forecast(train_series, valid_series, order)

    valid_metrics = calculate_metrics(valid_actual, valid_pred)
    return valid_pred, valid_actual, valid_metrics








# Final evaluation on the test set                    
# ---------------------------------------------------------------------------
# This function performs the final out-of-sample evaluation of the fixed
# ARIMA(p,d,q) configuration on completely unseen test data.
#
# The test evaluation represents the most important performance assessment
# because the model has not previously used these observations during
# training.
#
# Forecasting procedure:
#
# 1. Combine the training and validation series into one continuous
#    historical context dataset.
#
# 2. Apply the fixed 100-day sliding-window forecasting process
#    on the test set observations.
#
# 3. For each prediction step:
#       - fit the ARIMA model on the previous 100 observations
#       - generate a 1-step-ahead forecast (t+1 log return)
#
# 4. Compare predicted and actual log returns.
#
# 5. Compute the final evaluation metrics including:
#       - MAE
#       - RMSE
#       - MAPE
#       - Directional Accuracy
#       - R²
#       - Explained Variance
#
# Unlike traditional walk-forward validation, the historical window size
# remains fixed rather than continuously expanding. This maintains direct
# methodological consistency with the LSTM and CNN-LSTM evaluation pipelines.
#
# Returns:
# - test_pred    : predicted log returns for the test set
# - test_actual  : true log returns corresponding to the predictions
# - test_metrics : dictionary containing all evaluation metrics
def evaluate_on_test(train_series, valid_series, test_series, order):
    """
    Evaluate the fixed ARIMA order on the held-out test set using the same
    100-day lookback approach used during validation.

    Context for the first test window is the full train + validation series.
    No refitting on growing history — consistent with the LSTM methodology.
    """
    # Combine train and validation as the context window source for the test set
    train_valid_series = pd.concat([train_series, valid_series], axis=0)

    # ------------------------------------------------------------------
    # 100-day lookback on test set (replaces walk-forward eval)
    # ------------------------------------------------------------------
    test_pred, test_actual = lookback_forecast(train_valid_series, test_series, order)

    test_metrics = calculate_metrics(test_actual, test_pred)
    return test_pred, test_actual, test_metrics







# Output helpers     
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.              
# ---------------------------------------------------------------------------
# This function exports:
# - the final test predictions and actual values
# - a summary table containing train, validation, and test metrics
#
# The saved files allow further analysis, visualisation, and comparison
# of the ARIMA forecasting performance outside the Python environment.
def save_outputs(full_df, train_len, valid_len, test_pred, test_actual,
                 order, train_metrics, valid_metrics, test_metrics):
    """Save test predictions and summary metrics to CSV files."""

    # Build a predictions dataframe aligned to the test portion of the data
    start_idx = train_len + valid_len
    pred_df = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()
    pred_df["actual_log_return"]           = test_actual
    pred_df["predicted_log_return_t_plus_1"] = test_pred
    pred_df["arima_order"]                 = str(order)
    pred_df["lookback_window"]             = LOOKBACK
    pred_df["test_R2"]                     = test_metrics["R2"]
    pred_df["test_Explained_Variance"]     = test_metrics["Explained_Variance"]

    pred_df.to_csv("arima_test_predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {"dataset": "train", "arima_order": str(order), "lookback_window": LOOKBACK, **train_metrics},
        {"dataset": "validation", "arima_order": str(order), "lookback_window": LOOKBACK, **valid_metrics},
        {"dataset": "test", "arima_order": str(order), "lookback_window": LOOKBACK, **test_metrics},
    ])
    metrics_df.to_csv("arima_metrics_summary.csv", index=False)

    logging.info("Saved test predictions       -> arima_test_predictions.csv")
    logging.info("Saved metrics summary        -> arima_metrics_summary.csv")








# Main entry point      
# ---------------------------------------------------------------------------
# CLAUDE SONNET 4.6
# Some intermediate print statements were added with help from Claude to monitor runtime progress.           
# ---------------------------------------------------------------------------
# This function executes the complete ARIMA forecasting workflow:
# - load and preprocess the data
# - split the dataset into train/validation/test sets
# - evaluate the fixed ARIMA order on all datasets
# - print forecasting results and evaluation metrics
# - save outputs and predictions to CSV files
#
# The function serves as the central controller of the entire forecasting
# and evaluation process.
def arima_forecast():

    # Step 1: Load CSV and compute daily log returns from Close price
    df = load_data(CSV_PATH)

    # Step 2: Split into train / validation / test (70 / 15 / 15)
    train_df, valid_df, test_df = split_series(df)

    train_series = train_df["log_return"].reset_index(drop=True)
    valid_series = valid_df["log_return"].reset_index(drop=True)
    test_series  = test_df["log_return"].reset_index(drop=True)

    # Step 3: Use fixed ARIMA order — no hyperparameter search is performed.
    #         ARIMA(1, 0, 1) is a standard baseline for stationary return series.
    order = ARIMA_ORDER
    logging.info("Using fixed ARIMA order: %s", order)

    # Step 4a: Evaluate the fixed order on the training set (in-sample,
    #          100-day lookback starting from row LOOKBACK of the train set).
    train_pred, train_actual, train_metrics = evaluate_on_train(
        train_series, order
    )

    # Step 4b: Evaluate the fixed order on the validation set so we can
    #          report its metrics in the same structured format.
    valid_pred, valid_actual, valid_metrics = evaluate_on_validation(
        train_series, valid_series, order
    )

    # Step 4c: Evaluate the fixed order on the held-out test set using the same
    #          100-day lookback approach (replaces original walk-forward eval).
    test_pred, test_actual, test_metrics = evaluate_on_test(
        train_series, valid_series, test_series, order
    )

    # ------------------------------------------------------------------
    # Step 5: Print results in the same format as the XGBoost output
    # ------------------------------------------------------------------
    SEP = "=" * 55

    # ---- TRAIN ----
    print(f"\n{SEP}")
    print(f"  TRAIN RESULTS  -  ARIMA (t+1 forecast)")
    print(f"  [Before Hyperparameter Optimisation - Fixed Order]")
    print(SEP)
    print(f"  Params               : {order}")
    print(f"  Forecast horizon     : t+1")
    print(f"  MAE                  : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {train_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {train_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {train_metrics['Explained_Variance']:.8f}")
    print(SEP)

    # ---- VALIDATION ----
    print(f"\n{SEP}")
    print(f"  VALIDATION RESULTS  -  ARIMA (t+1 forecast)")
    print(f"  [Before Hyperparameter Optimisation - Fixed Order]")
    print(SEP)
    print(f"  Params               : {order}")
    print(f"  Forecast horizon     : t+1")
    print(f"  MAE                  : {valid_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {valid_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {valid_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {valid_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {valid_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {valid_metrics['Explained_Variance']:.8f}")
    print(SEP)

    # ---- TEST ----
    print(f"\n{SEP}")
    print(f"  FINAL TEST RESULTS  -  ARIMA (t+1 forecast)")
    print(f"  [Before Hyperparameter Optimisation - Fixed Order]")
    print(SEP)
    print(f"  Params               : {order}")
    print(f"  Forecast horizon     : t+1")
    print(f"  MAE                  : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {test_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {test_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {test_metrics['Explained_Variance']:.8f}")
    print(SEP)

    logging.info("----- TRAIN RESULTS (100-day lookback, in-sample) -----")
    logging.info("ARIMA order              : %s", order)
    logging.info("MAE                      : %.8f", train_metrics["MAE"])
    logging.info("RMSE                     : %.8f", train_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", train_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", train_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", train_metrics["R2"])
    logging.info("Explained Variance       : %.8f", train_metrics["Explained_Variance"])

    logging.info("----- VALIDATION RESULTS (100-day lookback) -----")
    logging.info("ARIMA order              : %s", order)
    logging.info("MAE                      : %.8f", valid_metrics["MAE"])
    logging.info("RMSE                     : %.8f", valid_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", valid_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", valid_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", valid_metrics["R2"])
    logging.info("Explained Variance       : %.8f", valid_metrics["Explained_Variance"])

    logging.info("----- FINAL TEST RESULTS (100-day lookback, out-of-sample) -----")
    logging.info("ARIMA order              : %s", order)
    logging.info("MAE                      : %.8f", test_metrics["MAE"])
    logging.info("RMSE                     : %.8f", test_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", test_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", test_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", test_metrics["R2"])
    logging.info("Explained Variance       : %.8f", test_metrics["Explained_Variance"])

    # Step 6: Save outputs to disk
    save_outputs(
        full_df       = df,
        train_len     = len(train_df),
        valid_len     = len(valid_df),
        test_pred     = test_pred,
        test_actual   = test_actual,
        order         = order,
        train_metrics = train_metrics,
        valid_metrics = valid_metrics,
        test_metrics  = test_metrics,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    arima_forecast()
