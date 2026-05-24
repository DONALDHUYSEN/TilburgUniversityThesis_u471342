"""
per_sample_shap_value_xgboost_1.py

Bereken per-sample SHAP values voor een getraind XGBoost-model.

Input-bestanden verwacht in dezelfde map als dit script, of geef ze mee via CLI:
- train_t1.csv
- test_t1.csv
- xgb_model_t1.json

Output:
- outputs/per_sample_shap_values_xgboost_1.csv
- outputs/per_sample_shap_values_normalized_xgboost_1.csv
- outputs/per_sample_abs_shap_values_normalized_xgboost_1.csv
- outputs/per_sample_top_features_xgboost_1.csv
- outputs/shap_feature_summary_xgboost_1.csv
- outputs/per_sample_shap_metadata_xgboost_1.json

Gebruik:
python per_sample_shap_value_xgboost_1.py

Optioneel:
python per_sample_shap_value_xgboost_1.py \
  --train train_t1.csv \
  --test test_t1.csv \
  --model xgb_model_t1.json \
  --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb


DEFAULT_EXCLUDE_COLUMNS = {
    "Date",
    "asset",
    "target",
    "target_log_return_t_plus_1",
    "log_return",
    "Close",
}

# Extra presets voor het geval het model met net een andere feature-set is getraind.
# Het script kiest automatisch de eerste preset waarvan het aantal features overeenkomt
# met het aantal features in het opgeslagen XGBoost-model.
EXCLUDE_PRESETS = [
    DEFAULT_EXCLUDE_COLUMNS,
    {"Date", "asset", "target_log_return_t_plus_1", "log_return"},
    {"Date", "asset", "target_log_return_t_plus_1", "Close"},
    {"Date", "asset", "target_log_return_t_plus_1"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bereken per-sample SHAP values voor XGBoost."
    )
    parser.add_argument("--train", default="train_t1.csv", help="Pad naar train CSV.")
    parser.add_argument("--test", default="test_t1.csv", help="Pad naar test CSV.")
    parser.add_argument("--model", default="xgb_model_t1.json", help="Pad naar XGBoost JSON model.")
    parser.add_argument("--output-dir", default="outputs", help="Map voor outputbestanden.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Aantal belangrijkste features per sample voor de top-features output.",
    )
    parser.add_argument(
        "--extra-exclude",
        nargs="*",
        default=[],
        help="Extra kolommen die expliciet uit X moeten worden verwijderd.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def get_model_num_features(booster: xgb.Booster) -> int:
    num_features = booster.num_features()
    if num_features <= 0:
        raise ValueError("Kon het aantal features in het XGBoost-model niet bepalen.")
    return num_features


def numeric_feature_candidates(df: pd.DataFrame, exclude_columns: Iterable[str]) -> list[str]:
    exclude = set(exclude_columns)
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return numeric_cols


def infer_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_num_features: int,
    extra_exclude: Iterable[str] = (),
) -> tuple[list[str], set[str]]:
    shared_columns = [c for c in train_df.columns if c in test_df.columns]
    train_shared = train_df[shared_columns]

    tried: list[tuple[set[str], int]] = []
    for preset in EXCLUDE_PRESETS:
        exclude = set(preset) | set(extra_exclude)
        features = numeric_feature_candidates(train_shared, exclude)
        tried.append((exclude, len(features)))
        if len(features) == expected_num_features:
            return features, exclude

    tried_msg = "\n".join(
        f"- exclude={sorted(list(exclude))}: {n_features} features"
        for exclude, n_features in tried
    )
    raise ValueError(
        "Geen feature-set gevonden die overeenkomt met het model.\n"
        f"Het model verwacht {expected_num_features} features.\n"
        "Geprobeerde presets:\n"
        f"{tried_msg}\n\n"
        "Oplossing: geef extra kolommen op via bijvoorbeeld:\n"
        "python per_sample_shap_value_xgboost_1.py --extra-exclude kolomnaam1 kolomnaam2"
    )


def make_meta_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    meta = pd.DataFrame(index=df.index)
    if "Date" in df.columns:
        meta["Date"] = df["Date"].values
    meta["split"] = split_name
    meta["sample_index_within_split"] = np.arange(len(df))
    return meta


def calculate_shap_contributions(booster: xgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dmatrix = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=X.columns.tolist())

    # pred_contribs=True geeft SHAP-contributies per sample per feature.
    # De laatste kolom is de bias/base value en hoort niet bij de features.
    contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    bias_values = contribs[:, -1]
    return shap_values, bias_values


def save_outputs(
    output_dir: Path,
    meta: pd.DataFrame,
    feature_names: list[str],
    shap_values: np.ndarray,
    bias_values: np.ndarray,
    train_rows: int,
    test_rows: int,
    top_k: int,
    selected_exclude_columns: set[str],
    input_paths: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    out_shap = pd.concat([meta.reset_index(drop=True), shap_df], axis=1)
    out_shap.to_csv(output_dir / "per_sample_shap_values_xgboost_1.csv", index=False)

    max_abs = float(np.nanmax(np.abs(shap_values)))
    if max_abs == 0 or np.isnan(max_abs):
        normalized = np.zeros_like(shap_values)
    else:
        normalized = shap_values / max_abs

    norm_df = pd.DataFrame(normalized, columns=feature_names)
    out_norm = pd.concat([meta.reset_index(drop=True), norm_df], axis=1)
    out_norm.to_csv(output_dir / "per_sample_shap_values_normalized_xgboost_1.csv", index=False)

    abs_norm_df = pd.DataFrame(np.abs(normalized), columns=feature_names)
    out_abs_norm = pd.concat([meta.reset_index(drop=True), abs_norm_df], axis=1)
    out_abs_norm.to_csv(output_dir / "per_sample_abs_shap_values_normalized_xgboost_1.csv", index=False)

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
            "std_shap": np.std(shap_values, axis=0),
            "max_abs_shap": np.max(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(output_dir / "shap_feature_summary_xgboost_1.csv", index=False)

    top_k = min(top_k, len(feature_names))
    abs_values = np.abs(shap_values)
    top_indices = np.argsort(-abs_values, axis=1)[:, :top_k]
    rows = []
    for i in range(shap_values.shape[0]):
        base = meta.iloc[i].to_dict()
        for rank, j in enumerate(top_indices[i], start=1):
            rows.append(
                {
                    **base,
                    "rank": rank,
                    "feature": feature_names[j],
                    "shap_value": shap_values[i, j],
                    "abs_shap_value": abs_values[i, j],
                    "normalized_shap_value": normalized[i, j],
                    "abs_normalized_shap_value": abs(normalized[i, j]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "per_sample_top_features_xgboost_1.csv", index=False)

    metadata = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "combined_rows": train_rows + test_rows,
        "train_test_split_index": train_rows,
        "num_features": len(feature_names),
        "selected_exclude_columns": sorted(selected_exclude_columns),
        "max_abs_shap_for_normalization": max_abs,
        "bias_value_mean": float(np.mean(bias_values)),
        "input_paths": input_paths,
        "outputs": {
            "raw_shap_values": "per_sample_shap_values_xgboost_1.csv",
            "normalized_shap_values": "per_sample_shap_values_normalized_xgboost_1.csv",
            "abs_normalized_shap_values": "per_sample_abs_shap_values_normalized_xgboost_1.csv",
            "top_features_per_sample": "per_sample_top_features_xgboost_1.csv",
            "feature_summary": "shap_feature_summary_xgboost_1.csv",
        },
    }
    with open(output_dir / "per_sample_shap_metadata_xgboost_1.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "feature_names_xgboost_1.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    booster = load_model(model_path)
    expected_num_features = get_model_num_features(booster)

    feature_names, selected_exclude_columns = infer_feature_columns(
        train_df=train_df,
        test_df=test_df,
        expected_num_features=expected_num_features,
        extra_exclude=args.extra_exclude,
    )

    X_train = train_df[feature_names].copy()
    X_test = test_df[feature_names].copy()
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # XGBoost kan missing values aan, maar object columns horen hier niet meer in te zitten.
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan)

    meta_train = make_meta_columns(train_df, "train")
    meta_test = make_meta_columns(test_df, "test")
    meta = pd.concat([meta_train, meta_test], axis=0, ignore_index=True)
    meta["sample_index_combined"] = np.arange(len(meta))

    shap_values, bias_values = calculate_shap_contributions(booster, X_combined)

    save_outputs(
        output_dir=output_dir,
        meta=meta,
        feature_names=feature_names,
        shap_values=shap_values,
        bias_values=bias_values,
        train_rows=len(train_df),
        test_rows=len(test_df),
        top_k=args.top_k,
        selected_exclude_columns=selected_exclude_columns,
        input_paths={
            "train": str(train_path),
            "test": str(test_path),
            "model": str(model_path),
        },
    )

    print("Klaar. SHAP outputs zijn opgeslagen in:", output_dir.resolve())
    print("Aantal train samples:", len(train_df))
    print("Aantal test samples:", len(test_df))
    print("Train/test split index:", len(train_df))
    print("Aantal features:", len(feature_names))
    print("Gebruikte exclude columns:", sorted(selected_exclude_columns))


if __name__ == "__main__":
    main()
