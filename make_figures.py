"""Figures for the MR-LSTM / MR-GRU report.

Print-targeted (static PDF), so no hover layer: identity is carried by direct
labels and by the tables in the report body. Emphasis form throughout -- the
two new models carry hue, the baselines recede to muted gray -- because the
story is "these two vs the baselines", not "five equal series".
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- design tokens (light surface; see dataviz references/palette.md) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1 = "#2a78d6"   # MR-LSTM
S2 = "#eb6834"   # MR-GRU
DEEMPH = "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
    "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

ORDER = ["Naive (persistence)", "MRINN (T=1)", "MRINN (T=8)",
         "MR-LSTM (T=8)", "MR-GRU (T=8)"]
COLOR = {"MR-LSTM (T=8)": S1, "MR-GRU (T=8)": S2}


def color_for(name):
    return COLOR.get(name, DEEMPH)


def load_results():
    df = pd.read_csv(HERE / "results" / "comparison_e50_T8.csv", index_col=0)
    return df.loc[[m for m in ORDER if m in df.index]]


def header(ax, title, subtitle=None, title_y=1.16, sub_y=1.05):
    """Title above subtitle, both in axes coords so they never collide."""
    ax.text(0, title_y, title, transform=ax.transAxes, fontsize=10.5,
            color=INK, fontweight="bold", va="bottom", ha="left")
    if subtitle:
        ax.text(0, sub_y, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTED, va="bottom", ha="left")


def style_axes(ax, xlabel=None):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)


def hbars(ax, names, values, fmt="{:.2f}", title="", subtitle=""):
    y = np.arange(len(names))[::-1]
    for yi, n, v in zip(y, names, values):
        ax.barh(yi, v, height=0.62, color=color_for(n),
                edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left",
                fontsize=8.5, color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(values) * 1.22)
    header(ax, title, subtitle)
    style_axes(ax)


def fig_metrics(df):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    specs = [("AQL", "Average Quantile Loss", "lower is better"),
             ("MAE", "Mean Absolute Error", "EUR/MWh, lower is better"),
             ("RMSE", "Root Mean Squared Error", "EUR/MWh, lower is better")]
    for ax, (col, title, sub) in zip(axes, specs):
        hbars(ax, list(df.index), df[col].to_numpy(), title=title, subtitle=sub)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "fig_metrics.png")
    plt.close(fig)


def fig_efficiency(df):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    d = df[df["Params"] > 0]
    nv = df.loc["Naive (persistence)", "AQL"]

    # The naive baseline (AQL 14.24) is far off this scale. Plotting it would
    # flatten the 12.20-12.45 spread that is the actual subject, so it is
    # stated in the subtitle instead of drawn.
    ax.set_xlim(1200, 6300)
    ax.set_ylim(12.14, 12.52)

    # label placement per point, to keep text clear of the axes and each other
    offsets = {"MRINN (T=1)": (0, 15, "left"), "MRINN (T=8)": (0, 15, "center"),
               "MR-LSTM (T=8)": (0, 15, "right"), "MR-GRU (T=8)": (-14, -38, "right")}
    for name, r in d.iterrows():
        big = name in COLOR
        dx, dy, ha = offsets.get(name, (0, 15, "center"))
        ax.scatter(r["Params"], r["AQL"], s=200 if big else 140,
                   color=color_for(name), edgecolor=SURFACE, linewidth=2,
                   zorder=4 if big else 3)
        ax.annotate(f"{name}\n{r['AQL']:.3f}", (r["Params"], r["AQL"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8.5, linespacing=1.45,
                    color=INK if big else INK_2,
                    fontweight="bold" if big else "normal")

    header(ax, "Accuracy vs. model size",
           f"AQL (lower is better) against trainable parameters  -  "
           f"naive baseline AQL {nv:.2f}, off scale",
           title_y=1.11, sub_y=1.035)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("AQL")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_efficiency.png")
    plt.close(fig)


def fig_calibration(df):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    hbars(axes[0], list(df.index), df["AQCE"].to_numpy(),
          title="AQCE  -  coverage error",
          subtitle="percentage points, lower is better")
    hbars(axes[1], list(df.index), df["AIW"].to_numpy(),
          title="AIW  -  interval width",
          subtitle="EUR/MWh, narrower = sharper")
    axes[1].set_yticklabels([])
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "fig_calibration.png")
    plt.close(fig)


def fig_forecast():
    """Fan chart from the trained MR-GRU checkpoint."""
    import library_mrrnn as MR
    from compare import (FEATS_CAPACITIES, FEATS_PRICES, FEATS_VOLUME, LABEL,
                         QUANTILES, prepare)

    feats = FEATS_PRICES + FEATS_CAPACITIES + FEATS_VOLUME
    data_path = str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv")
    regelzonen_data, _ = MR.load_data(
        FEATS_PRICES, FEATS_CAPACITIES, FEATS_VOLUME, LABEL, data_path)

    lags = list(range(1, 9))
    _, _, split = prepare(regelzonen_data, feats, lags)
    X_test, y_test, y_scaler = split[2], split[5], split[6]

    ckpt = str(HERE / "results" / "best_gru_T8.keras")
    yqs = MR.make_inference(X_test, lags, QUANTILES, checkpoint_path=ckpt)

    inv = lambda a: y_scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
    q = {qq: inv(a) for qq, a in zip(QUANTILES, yqs)}
    truth = inv(y_test.to_numpy())

    s, n = 2400, 336  # one week of 15-minute intervals
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.fill_between(x, q[0.1][s:s+n], q[0.9][s:s+n], color=S2, alpha=0.16,
                    linewidth=0, zorder=2, label="80% interval (q10-q90)")
    ax.fill_between(x, q[0.25][s:s+n], q[0.75][s:s+n], color=S2, alpha=0.28,
                    linewidth=0, zorder=3, label="50% interval (q25-q75)")
    ax.plot(x, q[0.5][s:s+n], color=S2, linewidth=1.8, zorder=5,
            label="MR-GRU median")
    ax.plot(x, truth[s:s+n], color=INK, linewidth=1.1, alpha=0.85, zorder=6,
            label="Actual")
    header(ax, "MR-GRU probabilistic forecast vs. actual imbalance price",
           "one week of the test set (336 x 15-minute intervals)",
           title_y=1.13, sub_y=1.04)
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("15-minute interval")
    ax.set_xlim(0, n - 1)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(length=0)
    leg = ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left",
                    bbox_to_anchor=(0, -0.22))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_forecast.png")
    plt.close(fig)


if __name__ == "__main__":
    df = load_results()
    fig_metrics(df)
    fig_efficiency(df)
    fig_calibration(df)
    print("metric figures written")
    fig_forecast()
    print("forecast figure written")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name, f"{p.stat().st_size//1024} KB")
