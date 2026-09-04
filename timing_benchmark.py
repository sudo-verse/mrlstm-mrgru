"""Computational-cost breakdown for the nine benchmarked models.

`benchmark_results.csv` records only `train_sec`, and that figure covers the WHOLE
per-fold block -- split, shift, scale, dataset build, fit, predict, inverse-transform
and scoring -- so it cannot be decomposed after the fact. This script measures the
components directly.

Method. Re-running every fit to completion would cost the original hours of compute for
timing alone, so the marginal per-epoch cost is measured on one representative fold and
scaled by the epoch counts already recorded.

The per-epoch cost is measured as a SLOPE, not as a single-epoch fit. A one-epoch `fit`
carries graph tracing and setup overhead that a fifty-epoch run amortises away, so
timing one epoch and multiplying overstates the total -- mildly for short runs and badly
for long ones (measured at +9% for MRINN at 53 epochs, +84% for LQR at 571). Timing two
fits of different length and differencing cancels that constant:

    per_epoch = (t(B epochs) - t(A epochs)) / (B - A)

Validation cost is isolated the same way, by differencing a fit WITH validation_data
against one without, so it reflects what the real run actually paid rather than a
standalone forward pass.

    training   = per-epoch slope, no validation   x (recorded epochs)
    validation = (with-val slope - no-val slope)  x (recorded epochs)
    testing    =  measured directly, one pass over the test set
    data prep  =  measured once; identical for every model of the same input layout

As a check, the reconstructed per-fold total is printed beside the recorded `train_sec`,
which covered prep + fit + predict + scoring together.

XGBoost has no epochs -- it fits 96 independent boosters with their own early stopping --
so its training column is a direct measurement of the full per-fold fit.

    .venv/bin/python timing_benchmark.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import library_mrrnn as MR                              # noqa: E402
from library_mrrnn import multihorizon as MH            # noqa: E402
from library_mrrnn import baselines_nn as B             # noqa: E402
from benchmark_models import (FEATS_PRICES, FEATS_CAP, FEATS_VOL, FEATS, LABEL,
                              T, LAGS, HU, HZ, Q, BATCH, BASE_LR, DA_DROP,
                              GATE_CLOSURE, SEED, STRIDE, MODELS, LABELS,
                              input_kind, seed_all)                  # noqa: E402

OUT = Path("results")
ORDER = ["MRINN", "MR-LSTM", "MR-GRU", "MLP", "LSTM", "GRU",
         "AttnBiLSTM", "XGBoost", "LQR"]


def timed(fn, repeat=1):
    """Wall time of `fn`, best of `repeat`, after one discarded warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


A_EP, B_EP = 2, 6          # slope endpoints for the per-epoch measurement


def epoch_slope(model, X, y, val=None):
    """Marginal seconds per epoch, with fixed setup overhead differenced out."""
    kw = {"batch_size": BATCH, "verbose": 0}
    if val is not None:
        kw["validation_data"] = val
    model.fit(X, y, epochs=1, **kw)                      # warm up / trace
    t0 = time.perf_counter(); model.fit(X, y, epochs=A_EP, **kw)
    ta = time.perf_counter() - t0
    t0 = time.perf_counter(); model.fit(X, y, epochs=B_EP, **kw)
    tb = time.perf_counter() - t0
    return max((tb - ta) / (B_EP - A_EP), 1e-6)


def main() -> None:
    res = pd.read_csv(OUT / "benchmark_results.csv")
    epochs = res.groupby("label").epochs.mean()
    recorded = res.groupby("label").train_sec.mean()

    data, _ = MR.load_data(FEATS_PRICES, FEATS_CAP, FEATS_VOL, LABEL,
                           str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv"))
    fold = MH.rolling_folds()[0]
    print(f"timing on fold {fold['tag']}, batch {BATCH}, T={T}, "
          f"{HZ} horizons, {len(Q)} quantiles\n")

    # build_dataset emits only two history layouts: flat (N,T) for "mrinn", and
    # (N,T,1) for everything else. input_kind() returns the CELL name for sequence
    # models, so collapse it to the layout that actually drives the cost.
    layout = lambda ik: "mrinn" if ik == "mrinn" else "gru"
    prep = {}
    sets = {}
    for ik in ("mrinn", "gru"):
        def _prep(ik=ik):
            tr, va, te = MR.split_data(data, fold["train"], fold["val"], fold["test"])
            trs, vas, tes, nm = MR.shift_data(tr, va, te, LABEL[0], FEATS, [], LAGS, [])
            Xtr, Xva, Xte, *_, ysc = MR.scale_data(trs, vas, tes, nm, LABEL)
            a = MH.build_dataset(tr, trs, Xtr, LAGS, ik, ysc, stride=STRIDE,
                                 gate_closure=GATE_CLOSURE)
            b = MH.build_dataset(va, vas, Xva, LAGS, ik, ysc, gate_closure=GATE_CLOSURE)
            c = MH.build_dataset(te, tes, Xte, LAGS, ik, ysc, gate_closure=GATE_CLOSURE)
            sets[ik] = (a, b, c, tr, FEATS_PRICES)
            return a
        prep[ik] = timed(_prep)
        (TRd, ytr, _), (VAd, yva, _), (TEd, yte, _), tr, _ = sets[ik]
        print(f"  data prep [{ik:5s}] {prep[ik]:6.1f} s   "
              f"train {len(ytr):,} / val {len(yva):,} / test {len(yte):,} origins")

    tr_split, _, _ = MR.split_data(data, fold["train"], fold["val"], fold["test"])
    C = list(MR.scaled_params(tr_split, FEATS_PRICES, FEATS_CAP)[:11])

    rows = []
    for name, (family, kind) in MODELS.items():
        label = LABELS[name]
        ik = layout(input_kind(family, kind))
        (TRd, ytr, _), (VAd, yva, _), (TEd, yte, _), _, _ = sets[ik]
        ep = float(epochs.get(label, np.nan))

        if family == "xgb":
            def _fit():
                for h in range(1, HZ + 1):
                    Xtr = B.xgb_features(TRd, h)
                    Xva = B.xgb_features(VAd, h)
                    B.fit_xgb_horizon(Xtr, ytr[:, h - 1], Xva, yva[:, h - 1], Q, seed=SEED)
            t_train = timed(_fit)          # measured in full: no epoch scaling
            def _pred():
                for h in range(1, HZ + 1):
                    B.xgb_features(TEd, h)
            t_test = timed(_pred)
            t_val, per_ep, per_val = float("nan"), float("nan"), float("nan")
        else:
            seed_all(SEED)
            model = (MH.build_multihorizon(kind, T, C, hidden_units=HU, horizons=HZ,
                                           quantiles=Q, lr=BASE_LR, da_dropout=DA_DROP)
                     if family == "mr" else
                     B.build_baseline(kind, T, hidden_units=HU, horizons=HZ,
                                      quantiles=Q, lr=BASE_LR, da_dropout=DA_DROP))
            per_ep = epoch_slope(model, TRd, ytr)
            per_both = epoch_slope(model, TRd, ytr, val=(VAd, yva))
            per_val = max(per_both - per_ep, 0.0)
            t_test = timed(lambda: model.predict(TEd, batch_size=1024, verbose=0),
                           repeat=2)
            t_train, t_val = per_ep * ep, per_val * ep
            tf.keras.backend.clear_session()

        total = np.nansum([prep[ik], t_train, t_val, t_test])
        rows.append({"model": label, "epochs": ep,
                     "prep_s": prep[ik], "s_per_epoch": per_ep,
                     "train_s": t_train, "val_s": t_val, "test_s": t_test,
                     "total_s": total, "recorded_s": float(recorded.get(label, np.nan))})
        print(f"  {label:<11s} epochs {ep:5.1f}  train {t_train:7.1f}  "
              f"val {t_val:6.1f}  test {t_test:5.2f}  total {total:7.1f} s  "
              f"(recorded {recorded.get(label, float('nan')):.0f})")

    t = pd.DataFrame(rows).set_index("model").reindex(ORDER).dropna(how="all")
    t.to_csv(OUT / "timing_breakdown.csv")
    print(f"\nwritten: {OUT / 'timing_breakdown.csv'}")
    print("\n  columns: prep_s and test_s are MEASURED; train_s and val_s are the")
    print("  measured per-epoch cost scaled by the recorded epoch count; recorded_s is")
    print("  the original per-fold wall time covering all of the above plus scoring.")


if __name__ == "__main__":
    main()
