import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates





#=======================
#  All code has been improved with CLAUDE-Sonnet-4.6
#================






# =========================
# SETTINGS
# =========================
path = "btc_full_feature_set_daily.csv"   # <-- pas aan
date_col_candidates = ["Date", "date", "timestamp", "Datetime", "Time"]  # probeert deze

# Zet hier kolommen die je NIET wilt plotten (optioneel)
exclude_cols = {
    "target", "future_return", "label"
}

# Output
out_file = "timeseries_all_features.png"   # of .jpg

# Layout
ncols = 8                  # veel kleine plots
fig_width_per_col = 3.0
fig_height_per_row = 2.2
dpi = 200

# =========================
# 1) Load data
# =========================
df = pd.read_csv(path)

# =========================
# 2) Find + parse date column
# =========================
date_col = None
for c in date_col_candidates:
    if c in df.columns:
        date_col = c
        break

if date_col is None:
    raise ValueError(f"Geen datumkolom gevonden. Verwacht iets als: {date_col_candidates}")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)

# =========================
# 3) Select numeric feature columns
# =========================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plot_cols = [c for c in num_cols if c not in exclude_cols]

if len(plot_cols) == 0:
    raise ValueError("Geen numerieke features om te plotten gevonden (na exclude).")

# =========================
# 4) Plot grid
# =========================
n = len(plot_cols)
nrows = math.ceil(n / ncols)

fig_w = ncols * fig_width_per_col
fig_h = nrows * fig_height_per_row

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=False)
axes = np.array(axes).reshape(-1)  # altijd 1D lijst

for i, col in enumerate(plot_cols):
    ax = axes[i]
    ax.plot(df[date_col], df[col], linewidth=1.0)
    # Jaar ticks instellen
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))   # elke 2 jaar (pas aan indien nodig)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis="x", rotation=45)
    
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Value", fontsize=7)

# Zet ongebruikte subplot-axes uit
for j in range(n, len(axes)):
    axes[j].axis("off")

fig.tight_layout()
fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_file}  | features plotted: {len(plot_cols)}")