"""Turn the benchmark CSVs into the tables the report needs.

Three outputs, in the order they should be read:

  1. THE GRID -- rules vs no-rules at matched encoder, paired by fold. This is the
     comparison the project has never run, and it is the one an examiner asks for first.
  2. THE HORIZON TABLE -- every model at 15 min, 1 hour and 1 day, read off the same
     multi-horizon forward pass.
  3. THE OVERALL TABLE -- horizon-averaged headline metrics against climatology.

Everything model-vs-model is reported PAIRED BY FOLD. Fold difficulty on this data spans
AQL 17 to 42 -- several times any plausible model effect -- so a pooled mean is dominated
by which quarters a model happened to be scored on. Pairing removes that entirely.

    .venv/bin/python summarise_benchmark.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results")
RES = OUT / "benchmark_results.csv"
HOR = OUT / "benchmark_horizon.csv"

# Free references, from baselines.py on the same rolling-fold row set.
CLIMATOLOGY = {"AQL": 29.150, "MAE": 75.204, "RMSE": 393.803}
SEASONAL_NAIVE = {"AQL": 39.870, "MAE": 96.027, "RMSE": 546.966}

TARGET_HORIZONS = [(1, "15 min"), (4, "1 hour"), (96, "1 day")]

# The 2 x 3 design: (encoder, with-rules model, rules-removed model)
GRID = [("Dense", "MRINN", "MLP"), ("LSTM", "MR-LSTM", "LSTM"), ("GRU", "MR-GRU", "GRU")]

ORDER = ["MRINN", "MR-LSTM", "MR-GRU", "MLP", "LSTM", "GRU",
         "AttnBiLSTM", "XGBoost", "LQR"]


def paired(d: pd.DataFrame, a: str, b: str, metric: str = "AQL"):
    """Mean difference a - b over folds both models completed. Negative favours `a`."""
    wide = d.pivot_table(index="fold", columns="label", values=metric)
    if a not in wide.columns or b not in wide.columns:
        return None
    both = wide[[a, b]].dropna()
    if both.empty:
        return None
    delta = both[a] - both[b]
    return {"n": len(delta), "mean": float(delta.mean()), "sd": float(delta.std(ddof=1))
            if len(delta) > 1 else float("nan"),
            "wins": int((delta < 0).sum()), "pct": float(delta.mean() / both[b].mean() * 100)}


def main() -> None:
    if not RES.exists():
        print(f"no results yet at {RES}")
        return
    d = pd.read_csv(RES)
    complete = d.groupby("label").fold.nunique()
    n_folds = int(d.fold.nunique())

    print(f"folds seen: {n_folds}   models: {len(complete)}")
    incomplete = complete[complete < n_folds]
    if len(incomplete):
        print("still running (partial): " + ", ".join(
            f"{k} {v}/{n_folds}" for k, v in incomplete.items()))
    print()

    # ---------------------------------------------------------------- 1. the grid
    print("=" * 78)
    print("1. WHAT DO THE MARKET RULES CONTRIBUTE?  (paired by fold, negative = rules help)")
    print("=" * 78)
    rows = []
    for enc, with_rules, without in GRID:
        for metric in ("AQL", "MAE"):
            p = paired(d, with_rules, without, metric)
            if p:
                rows.append({"encoder": enc, "comparison": f"{with_rules} - {without}",
                             "metric": metric, "folds": p["n"], "delta": p["mean"],
                             "pct": p["pct"], "rules_win": f"{p['wins']}/{p['n']}"})
    if rows:
        g = pd.DataFrame(rows)
        print(g.round(3).to_string(index=False))
        aql = g[g.metric == "AQL"]
        if len(aql):
            print(f"\n  across encoders, mean AQL effect of the rule blocks: "
                  f"{aql.delta.mean():+.3f} ({aql.pct.mean():+.2f}%)")
    else:
        print("  not enough models finished yet")

    # ------------------------------------------------------- 2. encoder comparisons
    print("\n" + "=" * 78)
    print("2. WHAT DOES THE ENCODER CONTRIBUTE?  (paired by fold, negative = first wins)")
    print("=" * 78)
    for a, b in [("MR-GRU", "MRINN"), ("MR-LSTM", "MRINN"), ("MR-GRU", "MR-LSTM"),
                 ("GRU", "MLP"), ("LSTM", "MLP"), ("GRU", "LSTM")]:
        p = paired(d, a, b)
        if p:
            print(f"  {a:>10s} - {b:<10s}  AQL {p['mean']:+7.3f}  ({p['pct']:+6.2f}%)  "
                  f"wins {p['wins']}/{p['n']} folds   sd {p['sd']:.3f}")

    # ----------------------------------------------------------- 3. overall table
    print("\n" + "=" * 78)
    print("3. OVERALL  (mean over 96 horizons; folds averaged)")
    print("=" * 78)
    agg = (d.groupby("label")
             .agg(folds=("fold", "nunique"), AQL=("AQL", "mean"), MAE=("MAE", "mean"),
                  RMSE=("RMSE", "mean"), AQCR=("AQCR", "max"),
                  cov80=("coverage80", "mean"), params=("params", "first"),
                  best_ep=("best_epoch", "mean"), mins=("train_sec", lambda s: s.sum() / 60))
             .reindex([m for m in ORDER if m in set(d.label)])
             .dropna(how="all"))
    agg["vs_clim_MAE_%"] = (agg.MAE / CLIMATOLOGY["MAE"] - 1) * 100
    agg["vs_clim_AQL_%"] = (agg.AQL / CLIMATOLOGY["AQL"] - 1) * 100
    # XGBoost has no weight count: `params` holds total boosting rounds summed over the
    # 96 per-horizon boosters. Printing that beside a neural weight count would invite a
    # false comparison, so blank it and say so.
    if "XGBoost" in agg.index:
        rounds = int(agg.loc["XGBoost", "params"])
        agg.loc["XGBoost", "params"] = float("nan")
        print(f"  note: XGBoost fits 96 boosters per fold ({rounds:,} boosting rounds "
              f"total); it has no comparable weight count, so `params` is left blank.")
    print(agg.round(3).to_string())
    print(f"\n  climatology     AQL {CLIMATOLOGY['AQL']:.3f}   MAE {CLIMATOLOGY['MAE']:.2f}"
          f"   RMSE {CLIMATOLOGY['RMSE']:.1f}")
    print(f"  seasonal naive  AQL {SEASONAL_NAIVE['AQL']:.3f}   MAE {SEASONAL_NAIVE['MAE']:.2f}"
          f"   RMSE {SEASONAL_NAIVE['RMSE']:.1f}")
    crossing = agg[agg.AQCR > 0]
    if len(crossing):
        print(f"\n  AQCR > 0 only for: {', '.join(crossing.index)}  "
              f"-- the models WITHOUT the hierarchical quantile head. "
              f"Zero elsewhere is earned by that head, not by the encoder.")

    # ------------------------------------------------------- 4. reporting horizons
    if HOR.exists():
        print("\n" + "=" * 78)
        print("4. THE THREE REPORTING HORIZONS  (folds averaged)")
        print("=" * 78)
        ph = pd.read_csv(HOR)
        lab = d[["model", "label"]].drop_duplicates()
        ph = ph.merge(lab, on="model", how="left")
        out = []
        for h, name in TARGET_HORIZONS:
            sub = (ph[ph.h == h].groupby("label")[["AQL", "MAE", "RMSE", "AQCR"]]
                   .mean().reindex([m for m in ORDER if m in set(ph.label)]).dropna(how="all"))
            print(f"\n--- {name} ahead (h={h}) ---")
            print(sub.round(3).to_string())
            for m, r in sub.iterrows():
                out.append({"horizon": name, "h": h, "model": m, **r.to_dict()})
        if out:
            path = OUT / "benchmark_by_horizon_summary.csv"
            pd.DataFrame(out).to_csv(path, index=False)
            print(f"\nwritten: {path}")

    # --------------------------------------------------- 5. POOLED, not fold-averaged
    # RMSE is a root of a mean of squares and does NOT average across folds: the mean of
    # four fold-RMSEs understates the pooled figure badly (321 vs ~389 here). MAE and AQL
    # are near-linear and survive averaging, RMSE does not. So pool the saved per-fold
    # predictions -- already inverted to EUR/MWh inside their own fold -- and score once.
    print("\n" + "=" * 78)
    print("5. POOLED OVER ALL FOLDS  (single evaluation on concatenated predictions)")
    print("=" * 78)
    name_of = dict(zip(d.model, d.label))
    pooled = []
    for model in d.model.unique():
        files = sorted(OUT.glob(f"pred_{model}_*.npz"))
        if len(files) < n_folds:
            continue
        yt = np.concatenate([np.load(f)["y_true"] for f in files])
        yp = np.concatenate([np.load(f)["y_pred"] for f in files])
        q50 = yp[:, :, 2]
        err = q50 - yt
        qs = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
        e = yt[:, :, None] - yp
        aql = float(np.mean(np.maximum(qs * e, (qs - 1.0) * e)))
        lo, hi = yp[:, :, 0], yp[:, :, -1]
        pooled.append({"model": name_of[model], "rows": len(yt), "AQL": aql,
                       "MAE": float(np.mean(np.abs(err))),
                       "RMSE": float(np.sqrt(np.mean(err ** 2))),
                       "cov80_%": float(np.mean((yt >= lo) & (yt <= hi)) * 100),
                       "AIW": float(np.mean(hi - lo)),
                       "extreme_preds": int((np.abs(q50) > 5000).sum())})
    if pooled:
        p = (pd.DataFrame(pooled).set_index("model")
             .reindex([m for m in ORDER if m in {r["model"] for r in pooled}]))
        print(p.round(3).to_string())
        print("\n  `extreme_preds` counts |q50| > 5000 EUR/MWh. MRINN has been observed to "
              "emit\n  tens of thousands of these under the expanding protocol; any "
              "non-zero count here\n  is worth inspecting before the number is reported.")
        p.to_csv(OUT / "benchmark_pooled.csv")
        print(f"\nwritten: {OUT / 'benchmark_pooled.csv'}")

    agg.to_csv(OUT / "benchmark_overall.csv")
    print(f"written: {OUT / 'benchmark_overall.csv'}")


if __name__ == "__main__":
    main()
