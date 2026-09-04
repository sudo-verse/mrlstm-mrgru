"""The results table, with the columns the supervisor asked for.

    MODEL | RULES | AQL | MAE | RMSE | sMAPE | MASE | AQCR | PARAMS | vs CLIM
          | train | validation | test | total computational time

Two things are done deliberately rather than conveniently.

POOLED, NOT FOLD-AVERAGED. RMSE is a root of a mean of squares and does not average
across folds -- the mean of four fold-RMSEs understates the pooled figure by ~17% on
this data. Every accuracy column is therefore computed once on the concatenated
per-fold predictions, each already inverted to EUR/MWh inside its own fold.

sMAPE IS REPORTED WITH ITS CAVEAT. On this series sMAPE is computable (the denominator
|y|+|y_hat| never vanishes) but it SATURATES: 6.8% of target cells are negative, and
8.8% of cells sit at the 200% ceiling where the metric is constant regardless of whether
the miss is 100 or 10,000 EUR/MWh. Those cells are 9.2% of the data but 29.6% of the
metric. sMAPE therefore cannot rank tail behaviour, which is what this market turns on.
MASE is printed beside it as the scale-free metric that does survive negative prices.

    .venv/bin/python results_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results")
QS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
CLIM = {"AQL": 29.150, "MAE": 75.204, "RMSE": 393.803}
ORDER = ["MRINN", "MR-LSTM", "MR-GRU", "MLP", "LSTM", "GRU",
         "AttnBiLSTM", "XGBoost", "LQR"]
# does the model carry the three differentiable market-rule blocks?
RULES = {"MRINN": "yes", "MR-LSTM": "yes", "MR-GRU": "yes", "MLP": "no",
         "LSTM": "no", "GRU": "no", "AttnBiLSTM": "no", "XGBoost": "no", "LQR": "no"}


def smape(y, p):
    den = np.abs(y) + np.abs(p)
    ok = den > 0
    return float(np.mean(200.0 * np.abs(p[ok] - y[ok]) / den[ok]))


def mase_scale(y, season=96):
    """Mean absolute seasonal-naive change, the MASE denominator."""
    return float(np.mean(np.abs(y[season:] - y[:-season])))


def main() -> None:
    res = pd.read_csv(OUT / "benchmark_results.csv")
    name_of = dict(zip(res.model, res.label))
    n_folds = res.fold.nunique()

    # a single scale for MASE, from the pooled targets of any completed model
    ref = sorted(OUT.glob("pred_mr_gru_*.npz"))
    scale = mase_scale(np.concatenate([np.load(f)["y_true"] for f in ref])[:, 0])

    rows = []
    for model in res.model.unique():
        files = sorted(OUT.glob(f"pred_{model}_*.npz"))
        if len(files) < n_folds:
            continue
        yt = np.concatenate([np.load(f)["y_true"] for f in files])
        yp = np.concatenate([np.load(f)["y_pred"] for f in files])
        q50 = yp[:, :, 2]
        err = q50 - yt
        e = yt[:, :, None] - yp
        lo, hi = yp[:, :, 0], yp[:, :, -1]
        sub = res[res.model == model]
        rows.append({
            "MODEL": name_of[model],
            "RULES": RULES.get(name_of[model], "-"),
            "AQL": float(np.mean(np.maximum(QS * e, (QS - 1.0) * e))),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE": smape(yt.ravel(), q50.ravel()),
            "MASE": float(np.mean(np.abs(err)) / scale),
            "AQCR": float(sub.AQCR.max()),
            "PARAMS": int(sub.params.iloc[0]),
            "cov80": float(np.mean((yt >= lo) & (yt <= hi)) * 100),
        })

    t = pd.DataFrame(rows).set_index("MODEL").reindex(
        [m for m in ORDER if m in {r["MODEL"] for r in rows}])
    t["vsCLIM_AQL_%"] = (t.AQL / CLIM["AQL"] - 1) * 100
    t["vsCLIM_MAE_%"] = (t.MAE / CLIM["MAE"] - 1) * 100

    tim = OUT / "timing_breakdown.csv"
    if tim.exists():
        tt = pd.read_csv(tim).set_index("model")
        for src, dst in [("train_s", "train_s"), ("val_s", "val_s"),
                         ("test_s", "test_s"), ("total_s", "total_s")]:
            t[dst] = tt[src].reindex(t.index)
    else:
        print("  (no results/timing_breakdown.csv yet -- run timing_benchmark.py)\n")

    # XGBoost's `params` holds boosting rounds, not weights: not comparable.
    if "XGBoost" in t.index:
        rounds = int(t.loc["XGBoost", "PARAMS"])
        t.loc["XGBoost", "PARAMS"] = np.nan
        note = (f"XGBoost fits 96 boosters per fold ({rounds:,} boosting rounds "
                f"total); it has no comparable weight count, so PARAMS is blank.")
    else:
        note = ""

    cols = ["RULES", "AQL", "MAE", "RMSE", "sMAPE", "MASE", "AQCR", "PARAMS",
            "vsCLIM_AQL_%", "vsCLIM_MAE_%"]
    if "total_s" in t.columns:
        t["total_min_4f"] = t.total_s * 4 / 60
        cols += ["train_s", "val_s", "test_s", "total_s", "total_min_4f"]
    print("=" * 128)
    print("RESULTS  (accuracy pooled over all folds: 34,521 origins x 96 horizons; "
          "time in seconds per fold)")
    print("=" * 128)
    print(t[cols].round(2).to_string())
    print(f"\n  climatology  AQL {CLIM['AQL']:.3f}  MAE {CLIM['MAE']:.2f}  "
          f"RMSE {CLIM['RMSE']:.1f}   (negative vs-CLIM = better than climatology)")
    if note:
        print(f"  note: {note}")
    print("  sMAPE saturates at 200% on 8.8% of cells and cannot rank the tail;")
    print("  MASE (seasonal-naive scaled, season=96) is the scale-free alternative.")

    if "total_s" in t.columns:
        print("  train_s/val_s are the measured per-epoch cost x the recorded epoch count;")
        print("  test_s and data prep are measured directly; total_min_4f is the whole")
        print("  four-fold study in minutes. XGBoost has no epochs, so its train_s is the")
        print("  full 96-booster fit measured directly and val_s does not apply.")

    t.to_csv(OUT / "results_table.csv")
    print(f"\nwritten: {OUT / 'results_table.csv'}")


if __name__ == "__main__":
    main()
