"""
V3_arima_btc_log_returns_random_search_7.py
============================================
ARIMA model for Bitcoin daily log-return forecasting (t+7 prediction).

Key changes from the t+1 version
----------------------------------
FORECAST HORIZON : Changed from t+1 to t+7. At each evaluation position the
           model is fitted on exactly the 100 most-recent observations that
           precede that point, then a 7-step-ahead forecast is produced.
           The prediction for step 7 is extracted and compared against the
           true log-return 7 days later. Eval positions advance one-by-one
           but each prediction looks 7 steps into the future.

ORIGINAL:  Rolling walk-forward evaluation — the model was re-fitted on an
           ever-growing history (full training set + every previously observed
           test point) and produced one step-ahead forecasts sequentially.

REVISED:   100-day fixed lookback window — at each evaluation position the
           model is fitted on exactly the 100 most-recent observations that
           precede that point, then a 7-step-ahead forecast is made.
           No rolling history accumulation occurs. This mirrors the
           sliding-window (lookback=100) approach used in the updated LSTM
           pipeline, ensuring methodological consistency across all models.

           Grid search over (p, d, q) has been replaced by a random search
           that samples exactly 60 unique ARIMA configurations from the same
           search space. Validation uses the 100-day lookback method. The best
           order is then evaluated on the test set the same way.
"""

import logging
import random
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "btc_clean.csv"
PRICE_COLUMN = "Close"
DATE_COLUMN = "Date"

# Train / validation / test proportions (unchanged from original)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

# ARIMA search space — same candidate ranges as the original grid search
P_VALUES = list(range(0, 6))   # 0, 1, 2, 3, 4, 5
D_VALUES = list(range(0, 3))   # 0, 1, 2
Q_VALUES = list(range(0, 6))   # 0, 1, 2, 3, 4, 5

# ---------------------------------------------------------------------------
# REVISED: fixed lookback window size (replaces rolling walk-forward history)
# ---------------------------------------------------------------------------
LOOKBACK = 100

# ---------------------------------------------------------------------------
# Forecast horizon: number of steps ahead to predict
# ---------------------------------------------------------------------------
FORECAST_HORIZON = 30

# ---------------------------------------------------------------------------
# Random search: how many unique ARIMA configurations to evaluate
# ---------------------------------------------------------------------------
# NOTE: Grid search (p x d x q = 6x3x6 = 108 combos) is replaced by random
#       search sampling 60 unique parameter combinations from the same space.
NUM_RANDOM_SEARCH_MODELS = 60

# Seed for reproducibility
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading & preprocessing        (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Train / validation / test split      (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metric helpers        (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# REVISED: 100-day lookback forecasting (replaces walk-forward evaluation) (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def lookback_forecast(context_series, eval_series, order, lookback=LOOKBACK,
                      horizon=FORECAST_HORIZON):
    """
    Fixed-window multi-step-ahead ARIMA forecasting.

    REPLACED LOGIC
    --------------
    Original walk-forward approach:
        - Fit once on the full training set.
        - After each prediction, append the true value so the history grows
          by one row at every step.

    NEW 100-day lookback approach (t+7)
    -------------------------------------
    For each position i in eval_series where i + horizon - 1 is still within
    eval_series:
        1. Build a window of exactly `lookback` observations ending just before
           position i. The first `lookback` rows come from context_series. As i
           advances further into eval_series, the window slides forward and
           includes more rows from eval_series (true values), always keeping
           exactly 100 obs.
        2. Fit a fresh ARIMA(order) model on that fixed-length window.
        3. Produce a `horizon`-step-ahead forecast and extract the final step
           (step 7), which is the t+7 log-return prediction.
        4. The corresponding actual is eval_series[i + horizon - 1].
        5. Discard the fitted model — no state is carried forward.

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
    horizon : int
        Number of steps ahead to forecast (default: FORECAST_HORIZON = 7).

    Returns
    -------
    predictions : np.ndarray  -- t+horizon forecasts
    actuals     : np.ndarray  -- corresponding true log-return values
    """

    # Concatenate context + eval into one continuous array for easy slicing
    full_series = pd.concat([context_series, eval_series], axis=0).reset_index(drop=True)
    context_len = len(context_series)

    predictions = []
    actuals     = []

    # We can only make a t+horizon prediction at position i if the target
    # (i + horizon - 1) still falls within eval_series.
    n_eval = len(eval_series)

    for step in range(n_eval - horizon + 1):
        # Index of the current "launch" position in full_series
        launch_idx = context_len + step

        # Index of the target observation (horizon steps ahead)
        target_idx = launch_idx + horizon - 1

        # Fixed lookback window: exactly the `lookback` rows preceding launch_idx
        window_start = launch_idx - lookback
        window_end   = launch_idx          # slice is exclusive at the end
        window = full_series.iloc[window_start:window_end].values

        if len(window) < lookback:
            logging.warning(
                "Window at step %d has only %d rows (expected %d). Skipping.",
                step, len(window), lookback,
            )
            continue

        # Fit ARIMA on the fixed lookback window and forecast `horizon` steps ahead
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model  = ARIMA(window, order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                # Extract the final step (t+horizon); handle Series and ndarray
                if hasattr(forecast, "iloc"):
                    pred_value = float(forecast.iloc[horizon - 1])
                else:
                    pred_value = float(forecast[horizon - 1])
            except Exception as exc:
                logging.warning("ARIMA%s failed at step %d: %s", order, step, exc)
                continue

        # Actual log-return at t+horizon
        actual_value = float(full_series.iloc[target_idx])

        predictions.append(pred_value)
        actuals.append(actual_value)

    return np.asarray(predictions, dtype=float), np.asarray(actuals, dtype=float)


# ---------------------------------------------------------------------------
# REVISED: Random search over (p, d, q) — replaces exhaustive grid search   (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def sample_unique_orders(n_samples, seed=RANDOM_SEED):
    """
    Sample up to n_samples unique (p, d, q) combinations from the same
    search space used in the original grid search.

    Duplicate draws are discarded so that we always test distinct configs.
    If the total search space is smaller than n_samples, all unique combos
    are returned instead (the space here has 6x3x6 = 108 unique combos so
    60 unique samples is always achievable).
    """
    random.seed(seed)

    all_possible_orders = []
    for p in P_VALUES:
        for d in D_VALUES:
            for q in Q_VALUES:
                all_possible_orders.append((p, d, q))

    # Shuffle and take the first n_samples unique combos
    random.shuffle(all_possible_orders)
    sampled = all_possible_orders[:n_samples]

    logging.info(
        "Total unique ARIMA configs in search space : %d",
        len(all_possible_orders),
    )
    logging.info(
        "Sampled configs for random search          : %d",
        len(sampled),
    )
    return sampled


def select_best_order(train_series, valid_series):
    """
    Random search over (p, d, q) space evaluated on the validation set.

    REPLACED LOGIC
    --------------
    Original: Exhaustive grid search — tested every combination in P x D x Q.

    NEW: Random search sampling exactly NUM_RANDOM_SEARCH_MODELS (60) unique
         ARIMA configurations from the same P_VALUES x D_VALUES x Q_VALUES
         space. Duplicates are discarded before testing so we always compare
         60 distinct models (or fewer if the space itself is smaller, which
         it is not here: 108 combos > 60).

         Each candidate is evaluated using the 100-day lookback approach
         (not walk-forward) on the validation set. Selection criterion is RMSE.
    """
    logging.info(
        "Starting random search on validation set "
        "(lookback=%d, n_models=%d, seed=%d)",
        LOOKBACK, NUM_RANDOM_SEARCH_MODELS, RANDOM_SEED,
    )

    # Sample 60 unique (p, d, q) combos from the original search space
    orders_to_test = sample_unique_orders(NUM_RANDOM_SEARCH_MODELS)

    results = []

    for order in orders_to_test:
        p, d, q = order
        try:
            # ------------------------------------------------------------------
            # REVISED: use 100-day lookback evaluation on validation set
            # (original used walk-forward evaluation here)
            # ------------------------------------------------------------------
            pred, actual = lookback_forecast(train_series, valid_series, order)

            if len(pred) == 0:
                logging.warning(
                    "ARIMA%s produced no valid predictions — skipping.", order
                )
                continue

            metrics = calculate_metrics(actual, pred)

            results.append({
                "p": p,
                "d": d,
                "q": q,
                "validation_rmse": metrics["RMSE"],
                "validation_mae":  metrics["MAE"],
                "validation_mape": metrics["MAPE"],
                "validation_directional_accuracy": metrics["Directional_Accuracy"],
                "validation_r2": metrics["R2"],
                "validation_explained_variance": metrics["Explained_Variance"],
            })

            logging.info(
                "ARIMA%s | valid RMSE=%.8f | MAE=%.8f | MAPE=%.4f | DA=%.2f%% | R2=%.8f | EV=%.8f",
                order,
                metrics["RMSE"],
                metrics["MAE"],
                metrics["MAPE"],
                metrics["Directional_Accuracy"],
                metrics["R2"],
                metrics["Explained_Variance"],
            )

        except Exception as exc:
            logging.warning("ARIMA%s failed: %s", order, exc)

    if not results:
        raise RuntimeError(
            "No ARIMA configuration could be fit successfully during random search."
        )

    # Sort by validation RMSE and pick the best
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("validation_rmse").reset_index(drop=True)

    best_row   = results_df.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))

    logging.info(
        "Best order selected: ARIMA%s  (validation RMSE=%.8f)",
        best_order, best_row["validation_rmse"],
    )
    return best_order, results_df


# ---------------------------------------------------------------------------
# Evaluation on the train set (in-sample, 100-day lookback)    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def evaluate_on_train(train_series, order):
    """
    Evaluate the best ARIMA order on the training set using the same
    100-day lookback approach.

    The first LOOKBACK rows are used as the initial context window.
    Predictions start from index LOOKBACK onward (train_series[LOOKBACK:]).
    Each prediction targets t+FORECAST_HORIZON, so the last
    FORECAST_HORIZON-1 rows of the eval portion have no matching actual
    and are automatically excluded by lookback_forecast.
    """
    # Context: the first LOOKBACK rows of train
    # Eval   : everything after that in the train set
    context = train_series.iloc[:LOOKBACK].reset_index(drop=True)
    eval_   = train_series.iloc[LOOKBACK:].reset_index(drop=True)

    train_pred, train_actual = lookback_forecast(context, eval_, order)

    train_metrics = calculate_metrics(train_actual, train_pred)
    return train_pred, train_actual, train_metrics


# ---------------------------------------------------------------------------
# Evaluation on the validation set with the best order    (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def evaluate_on_validation(train_series, valid_series, order):
    """
    Re-run the 100-day lookback forecast on the validation set using the
    best order selected by the random search.  This produces the validation
    metrics in the same format as the train and test result blocks.
    """
    valid_pred, valid_actual = lookback_forecast(train_series, valid_series, order)

    valid_metrics = calculate_metrics(valid_actual, valid_pred)
    return valid_pred, valid_actual, valid_metrics


# ---------------------------------------------------------------------------
# Final evaluation on the test set            (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def evaluate_on_test(train_series, valid_series, test_series, order):
    """
    Evaluate the best ARIMA order on the held-out test set using the same
    100-day lookback approach used during validation.

    Context for the first test window is the full train + validation series.
    No refitting on growing history — consistent with the LSTM methodology.
    """
    # Combine train and validation as the context window source for the test set
    train_valid_series = pd.concat([train_series, valid_series], axis=0)

    # ------------------------------------------------------------------
    # REVISED: 100-day lookback on test set (replaces walk-forward eval)
    # ------------------------------------------------------------------
    test_pred, test_actual = lookback_forecast(train_valid_series, test_series, order)

    test_metrics = calculate_metrics(test_actual, test_pred)
    return test_pred, test_actual, test_metrics


# ---------------------------------------------------------------------------
# Output helpers                (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def save_outputs(full_df, train_len, valid_len, test_pred, test_actual,
                 best_order, search_results, train_metrics, valid_metrics, test_metrics):
    """Save random search results, test predictions, and summary metrics to CSV files."""

    # Save the random search results table
    search_results.to_csv("arima_random_search_results_7.csv", index=False)

    # Build a predictions dataframe aligned to the test portion of the data.
    # Predictions start at test row 0 but target test row FORECAST_HORIZON-1,
    # so we align the output rows to the target positions.
    start_idx = train_len + valid_len + FORECAST_HORIZON - 1
    pred_df = full_df.iloc[start_idx : start_idx + len(test_pred)].copy()
    pred_df["actual_log_return"]                              = test_actual
    pred_df[f"predicted_log_return_t_plus_{FORECAST_HORIZON}"] = test_pred
    pred_df["best_order"]                                     = str(best_order)
    pred_df["lookback_window"]                                = LOOKBACK
    pred_df["forecast_horizon"]                               = FORECAST_HORIZON
    pred_df["test_R2"]                                        = test_metrics["R2"]
    pred_df["test_Explained_Variance"]                        = test_metrics["Explained_Variance"]

    pred_df.to_csv("arima_test_predictions_7.csv", index=False)

    metrics_df = pd.DataFrame([
        {"dataset": "train",      "best_order": str(best_order), "lookback_window": LOOKBACK, "forecast_horizon": FORECAST_HORIZON, **train_metrics},
        {"dataset": "validation", "best_order": str(best_order), "lookback_window": LOOKBACK, "forecast_horizon": FORECAST_HORIZON, **valid_metrics},
        {"dataset": "test",       "best_order": str(best_order), "lookback_window": LOOKBACK, "forecast_horizon": FORECAST_HORIZON, **test_metrics},
    ])
    metrics_df.to_csv("arima_metrics_summary_7.csv", index=False)

    logging.info("Saved random search results  -> arima_random_search_results_7.csv")
    logging.info("Saved test predictions       -> arima_test_predictions_7.csv")
    logging.info("Saved metrics summary        -> arima_metrics_summary_7.csv")


# ---------------------------------------------------------------------------
# Main entry point                   (IMPROVED BY CLAUDE-Sonnet-4.6)
# ---------------------------------------------------------------------------

def arima_forecast():

    # Step 1: Load CSV and compute daily log returns from Close price
    df = load_data(CSV_PATH)

    # Step 2: Split into train / validation / test (70 / 15 / 15)
    train_df, valid_df, test_df = split_series(df)

    train_series = train_df["log_return"].reset_index(drop=True)
    valid_series = valid_df["log_return"].reset_index(drop=True)
    test_series  = test_df["log_return"].reset_index(drop=True)

    # Step 3: Random search — sample 60 unique (p, d, q) configs and pick the
    #         best one based on validation RMSE using the 100-day lookback method.
    #         NOTE: this replaces the original exhaustive grid search.
    best_order, search_results = select_best_order(train_series, valid_series)

    # Step 4a: Evaluate the best order on the training set (in-sample,
    #          100-day lookback starting from row LOOKBACK of the train set).
    train_pred, train_actual, train_metrics = evaluate_on_train(
        train_series, best_order
    )

    # Step 4b: Re-evaluate the best order on the validation set so we can
    #          report its metrics in the same structured format.
    valid_pred, valid_actual, valid_metrics = evaluate_on_validation(
        train_series, valid_series, best_order
    )

    # Step 4c: Evaluate the best order on the held-out test set using the same
    #          100-day lookback approach (replaces original walk-forward eval).
    test_pred, test_actual, test_metrics = evaluate_on_test(
        train_series, valid_series, test_series, best_order
    )

    # ------------------------------------------------------------------
    # Step 5: Print results in the same format as the XGBoost output 
    # ------------------------------------------------------------------
    SEP = "=" * 55

    # ---- TRAIN ----
    print(f"\n{SEP}")
    print(f"  TRAIN RESULTS  -  ARIMA (t+{FORECAST_HORIZON} forecast)")
    print(f"  [After Hyperparameter Optimisation - Random Search]")
    print(SEP)
    print(f"  Best params          : {best_order}")
    print(f"  Forecast horizon     : t+{FORECAST_HORIZON}")
    print(f"  MAE                  : {train_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {train_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {train_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {train_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {train_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {train_metrics['Explained_Variance']:.8f}")
    print(SEP)

    # ---- VALIDATION ----
    print(f"\n{SEP}")
    print(f"  VALIDATION RESULTS  -  ARIMA (t+{FORECAST_HORIZON} forecast)")
    print(f"  [After Hyperparameter Optimisation - Random Search]")
    print(SEP)
    print(f"  Best params          : {best_order}")
    print(f"  Forecast horizon     : t+{FORECAST_HORIZON}")
    print(f"  MAE                  : {valid_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {valid_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {valid_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {valid_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {valid_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {valid_metrics['Explained_Variance']:.8f}")
    print(SEP)

    # ---- TEST ----
    print(f"\n{SEP}")
    print(f"  FINAL TEST RESULTS  -  ARIMA (t+{FORECAST_HORIZON} forecast)")
    print(f"  [After Hyperparameter Optimisation - Random Search]")
    print(SEP)
    print(f"  Best params          : {best_order}")
    print(f"  Forecast horizon     : t+{FORECAST_HORIZON}")
    print(f"  MAE                  : {test_metrics['MAE']:.8f}")
    print(f"  RMSE                 : {test_metrics['RMSE']:.8f}")
    print(f"  MAPE                 : {test_metrics['MAPE']:.4f}")
    print(f"  Directional Accuracy : {test_metrics['Directional_Accuracy']:.2f}%")
    print(f"  R^2                  : {test_metrics['R2']:.8f}")
    print(f"  Explained Variance   : {test_metrics['Explained_Variance']:.8f}")
    print(SEP)

    logging.info("----- TRAIN RESULTS (100-day lookback, in-sample, t+%d) -----", FORECAST_HORIZON)
    logging.info("Best ARIMA order         : %s", best_order)
    logging.info("MAE                      : %.8f", train_metrics["MAE"])
    logging.info("RMSE                     : %.8f", train_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", train_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", train_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", train_metrics["R2"])
    logging.info("Explained Variance       : %.8f", train_metrics["Explained_Variance"])

    logging.info("----- VALIDATION RESULTS (100-day lookback, t+%d) -----", FORECAST_HORIZON)
    logging.info("Best ARIMA order         : %s", best_order)
    logging.info("MAE                      : %.8f", valid_metrics["MAE"])
    logging.info("RMSE                     : %.8f", valid_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", valid_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", valid_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", valid_metrics["R2"])
    logging.info("Explained Variance       : %.8f", valid_metrics["Explained_Variance"])

    logging.info("----- FINAL TEST RESULTS (100-day lookback, out-of-sample, t+%d) -----", FORECAST_HORIZON)
    logging.info("Best ARIMA order         : %s", best_order)
    logging.info("MAE                      : %.8f", test_metrics["MAE"])
    logging.info("RMSE                     : %.8f", test_metrics["RMSE"])
    logging.info("MAPE                     : %.4f", test_metrics["MAPE"])
    logging.info("Directional Accuracy     : %.2f%%", test_metrics["Directional_Accuracy"])
    logging.info("R2                       : %.8f", test_metrics["R2"])
    logging.info("Explained Variance       : %.8f", test_metrics["Explained_Variance"])

    # Step 6: Save outputs to disk
    save_outputs(
        full_df        = df,
        train_len      = len(train_df),
        valid_len      = len(valid_df),
        test_pred      = test_pred,
        test_actual    = test_actual,
        best_order     = best_order,
        search_results = search_results,
        train_metrics  = train_metrics,
        valid_metrics  = valid_metrics,
        test_metrics   = test_metrics,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    arima_forecast()
