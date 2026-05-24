from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import joblib


HORIZON = 7

DEFAULT_EXCLUDE_COLUMNS = {
    "Date",
    "asset",
    "target",
    "target_log_return_t_plus_1",
    "target_log_return_t_plus_7",
    "log_return",
    "Close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bereken per-sample SHAP values voor LSTM t+7."
    )

    parser.add_argument("--train", default="train_t7.csv")
    parser.add_argument("--test", default="test_t7.csv")
    parser.add_argument("--model", default="lstm_model_t7.keras")
    parser.add_argument("--scaler", default="scaler_tv_t7.joblib")
    parser.add_argument("--output-dir", default="outputs_lstm_t7")

    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--explain-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--extra-exclude", nargs="*", default=[])

    return parser.parse_args()


def numeric_feature_candidates(
    df: pd.DataFrame,
    exclude_columns: Iterable[str],
) -> list[str]:
    exclude = set(exclude_columns)
    cols = [c for c in df.columns if c not in exclude]
    return df[cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()


def make_sequences(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    lookback: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    X_seq = []
    meta_rows = []

    for i in range(lookback - 1, len(X)):
        X_seq.append(X[i - lookback + 1 : i + 1])
        meta_rows.append(meta_df.iloc[i].to_dict())

    return np.asarray(X_seq, dtype=np.float32), pd.DataFrame(meta_rows)


def make_meta_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    meta = pd.DataFrame(index=df.index)

    if "Date" in df.columns:
        meta["Date"] = df["Date"].values

    meta["split"] = split_name
    meta["sample_index_within_split"] = np.arange(len(df))

    return meta


def normalize_shap(shap_values: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.nanmax(np.abs(shap_values)))

    if max_abs == 0 or np.isnan(max_abs):
        return np.zeros_like(shap_values), max_abs

    return shap_values / max_abs, max_abs


def get_model_input_shape(model) -> tuple[int, int]:
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    lookback = input_shape[1]
    n_features = input_shape[2]

    if lookback is None or n_features is None:
        raise ValueError(f"Kan input shape niet bepalen: {input_shape}")

    return int(lookback), int(n_features)


def calculate_shap_in_batches(
    explainer,
    X_explain: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    shap_batches = []

    for start in range(0, len(X_explain), batch_size):
        end = min(start + batch_size, len(X_explain))
        X_batch = X_explain[start:end]

        values = explainer.shap_values(X_batch)

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values)

        if values.ndim == 4:
            values = values[..., 0]

        shap_batches.append(values)

        print(f"SHAP batch klaar: {start} t/m {end}")

    return np.concatenate(shap_batches, axis=0)


def save_outputs(
    output_dir: Path,
    meta: pd.DataFrame,
    feature_names: list[str],
    shap_feature_values: np.ndarray,
    shap_feature_values_normalized: np.ndarray,
    top_k: int,
    train_sequence_rows: int,
    test_sequence_rows: int,
    lookback: int,
    max_abs_shap: float,
    input_paths: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_name = f"per_sample_shap_values_lstm_{HORIZON}.csv"
    norm_name = f"per_sample_shap_values_normalized_lstm_{HORIZON}.csv"
    abs_norm_name = f"per_sample_abs_shap_values_normalized_lstm_{HORIZON}.csv"
    top_name = f"per_sample_top_features_lstm_{HORIZON}.csv"
    summary_name = f"shap_feature_summary_lstm_{HORIZON}.csv"
    metadata_name = f"per_sample_shap_metadata_lstm_{HORIZON}.json"
    feature_names_file = f"feature_names_lstm_{HORIZON}.json"

    raw_df = pd.DataFrame(shap_feature_values, columns=feature_names)
    out_raw = pd.concat([meta.reset_index(drop=True), raw_df], axis=1)
    out_raw.to_csv(output_dir / raw_name, index=False)

    norm_df = pd.DataFrame(shap_feature_values_normalized, columns=feature_names)
    out_norm = pd.concat([meta.reset_index(drop=True), norm_df], axis=1)
    out_norm.to_csv(output_dir / norm_name, index=False)

    abs_norm_df = pd.DataFrame(
        np.abs(shap_feature_values_normalized),
        columns=feature_names,
    )
    out_abs_norm = pd.concat([meta.reset_index(drop=True), abs_norm_df], axis=1)
    out_abs_norm.to_csv(output_dir / abs_norm_name, index=False)

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(shap_feature_values), axis=0),
            "mean_shap": np.mean(shap_feature_values, axis=0),
            "std_shap": np.std(shap_feature_values, axis=0),
            "max_abs_shap": np.max(np.abs(shap_feature_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    summary.to_csv(output_dir / summary_name, index=False)

    top_k = min(top_k, len(feature_names))
    abs_values = np.abs(shap_feature_values)
    top_indices = np.argsort(-abs_values, axis=1)[:, :top_k]

    rows = []

    for i in range(shap_feature_values.shape[0]):
        base = meta.iloc[i].to_dict()

        for rank, j in enumerate(top_indices[i], start=1):
            rows.append(
                {
                    **base,
                    "rank": rank,
                    "feature": feature_names[j],
                    "shap_value": shap_feature_values[i, j],
                    "abs_shap_value": abs_values[i, j],
                    "normalized_shap_value": shap_feature_values_normalized[i, j],
                    "abs_normalized_shap_value": abs(shap_feature_values_normalized[i, j]),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / top_name, index=False)

    metadata = {
        "model_type": "LSTM",
        "horizon": HORIZON,
        "lookback": lookback,
        "train_sequence_rows": train_sequence_rows,
        "test_sequence_rows": test_sequence_rows,
        "combined_sequence_rows": train_sequence_rows + test_sequence_rows,
        "train_test_split_index": train_sequence_rows,
        "num_features": len(feature_names),
        "max_abs_shap_for_normalization": max_abs_shap,
        "input_paths": input_paths,
        "outputs": {
            "raw_shap_values": raw_name,
            "normalized_shap_values": norm_name,
            "abs_normalized_shap_values": abs_norm_name,
            "top_features_per_sample": top_name,
            "feature_summary": summary_name,
            "feature_names": feature_names_file,
        },
    }

    with open(output_dir / metadata_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / feature_names_file, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    output_dir = Path(args.output_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    model = tf.keras.models.load_model(model_path)

    model_lookback, model_num_features = get_model_input_shape(model)

    if args.lookback != model_lookback:
        print(
            f"Let op: opgegeven lookback={args.lookback}, "
            f"maar model verwacht lookback={model_lookback}. "
            f"Ik gebruik {model_lookback}."
        )
        args.lookback = model_lookback

    exclude_columns = DEFAULT_EXCLUDE_COLUMNS | set(args.extra_exclude)

    shared_columns = [c for c in train_df.columns if c in test_df.columns]
    train_shared = train_df[shared_columns]
    test_shared = test_df[shared_columns]

    feature_names = numeric_feature_candidates(train_shared, exclude_columns)

    if len(feature_names) != model_num_features:
        raise ValueError(
            f"Aantal gevonden features ({len(feature_names)}) komt niet overeen "
            f"met model input features ({model_num_features})."
        )

    X_train_raw = train_shared[feature_names].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test_shared[feature_names].replace([np.inf, -np.inf], np.nan)

    X_train_raw = X_train_raw.fillna(X_train_raw.median())
    X_test_raw = X_test_raw.fillna(X_train_raw.median())

    scaler = joblib.load(scaler_path)

    X_train_scaled = scaler.transform(X_train_raw.values)
    X_test_scaled = scaler.transform(X_test_raw.values)

    meta_train = make_meta_columns(train_df, "train")
    meta_test = make_meta_columns(test_df, "test")

    X_train_seq, meta_train_seq = make_sequences(
        X=X_train_scaled,
        meta_df=meta_train,
        lookback=args.lookback,
    )

    X_test_seq, meta_test_seq = make_sequences(
        X=X_test_scaled,
        meta_df=meta_test,
        lookback=args.lookback,
    )

    X_combined_seq = np.concatenate([X_train_seq, X_test_seq], axis=0)
    meta_combined = pd.concat([meta_train_seq, meta_test_seq], axis=0, ignore_index=True)
    meta_combined["sample_index_combined"] = np.arange(len(meta_combined))

    if args.explain_size is not None:
        X_explain = X_combined_seq[: args.explain_size]
        meta_explain = meta_combined.iloc[: args.explain_size].reset_index(drop=True)
    else:
        X_explain = X_combined_seq
        meta_explain = meta_combined

    background_size = min(args.background_size, len(X_train_seq))
    background = X_train_seq[:background_size]

    print("Model input shape:", model.input_shape)
    print("X_explain shape:", X_explain.shape)
    print("Background shape:", background.shape)

    explainer = shap.GradientExplainer(model, background)

    shap_values = calculate_shap_in_batches(
        explainer=explainer,
        X_explain=X_explain,
        batch_size=args.batch_size,
    )

    if shap_values.ndim != 3:
        raise ValueError(
            f"Onverwachte SHAP shape: {shap_values.shape}. "
            "Verwacht: samples x lookback x features."
        )

    shap_feature_values = shap_values.sum(axis=1)

    shap_feature_values_normalized, max_abs_shap = normalize_shap(shap_feature_values)

    save_outputs(
        output_dir=output_dir,
        meta=meta_explain,
        feature_names=feature_names,
        shap_feature_values=shap_feature_values,
        shap_feature_values_normalized=shap_feature_values_normalized,
        top_k=args.top_k,
        train_sequence_rows=len(X_train_seq),
        test_sequence_rows=len(X_test_seq),
        lookback=args.lookback,
        max_abs_shap=max_abs_shap,
        input_paths={
            "train": str(train_path),
            "test": str(test_path),
            "model": str(model_path),
            "scaler": str(scaler_path),
        },
    )

    print("Klaar. LSTM SHAP outputs zijn opgeslagen in:", output_dir.resolve())
    print("Aantal features:", len(feature_names))
    print("Lookback:", args.lookback)
    print("Train sequences:", len(X_train_seq))
    print("Test sequences:", len(X_test_seq))
    print("Train/test split index:", len(X_train_seq))
    print("Max abs SHAP voor normalisatie:", max_abs_shap)


if __name__ == "__main__":
    main()