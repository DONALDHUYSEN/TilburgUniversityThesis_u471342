import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# === Paths ===
shap_path = "per_sample_shap_values_normalized_cnnlstm_1.csv"
feature_names_path = "feature_names_cnnlstm_1.json"
metadata_path = "per_sample_shap_metadata_cnnlstm_1.json"

output_path = "V2_normalized_per_sample_shap_heatmap_cnnlstm_1.png"

# === Load files ===
shap_df = pd.read_csv(shap_path)

with open(feature_names_path, "r") as f:
    feature_names = json.load(f)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

split_index = metadata["train_test_split_index"]

# === Optional: remove lagged features for readability ===
plot_features = [
    f for f in feature_names
    if "_lag" not in f and not f.startswith("lag_") and f != "Close_lag1"
]

# Keep only selected features that exist in SHAP file
plot_features = [f for f in plot_features if f in shap_df.columns]

# === Matrix: rows = features, columns = samples ===
heatmap_data = shap_df[plot_features].T.values

# === Sort features by TRAIN-only mean absolute importance ===
train_data = heatmap_data[:, :split_index]
mean_abs_importance = np.mean(np.abs(train_data), axis=1)
sort_idx = np.argsort(mean_abs_importance)[::-1]

heatmap_data = heatmap_data[sort_idx]
plot_features = [plot_features[i] for i in sort_idx]

# === Log transform (signed) ===
heatmap_log = np.sign(heatmap_data) * np.log1p(np.abs(heatmap_data))

# === Scale anchored to train 99th percentile ===
train_log = heatmap_log[:, :split_index]
scale = np.percentile(np.abs(train_log), 99)
scale = max(scale, 1e-8)

# === Plot ===
plt.figure(figsize=(18, max(8, len(plot_features) * 0.16)))

# TwoSlopeNorm keeps zero anchored at the colormap center
norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)

im = plt.imshow(
    heatmap_log,
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest",
    norm=norm,
)

plt.axvline(
    x=split_index - 0.5,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Train/Test Split"
)

plt.yticks(
    ticks=np.arange(len(plot_features)),
    labels=plot_features,
    fontsize=7
)

plt.xlabel("Samples (Train | Test)")
plt.ylabel("Features")
plt.title("CNN-LSTM: Normalized Feature Importance Over Combined Train and Test Samples t+1")

cbar = plt.colorbar(im)
cbar.set_label("SHAP value (log-scaled color, original values on ticks)")

log_ticks = np.linspace(-scale, scale, 9)
original_ticks = np.sign(log_ticks) * (np.expm1(np.abs(log_ticks)))
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f"{v:.3f}" for v in original_ticks], fontsize=7)

plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {output_path}")
