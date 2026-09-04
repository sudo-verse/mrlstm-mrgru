"""Nine models, one recipe, three reporting horizons -- the comparison table.

Every model is trained on identical rolling folds with identical scaling, identical
known-future covariates, identical `DAOutageDropout` and -- crucially -- the
gate-closure fix ON, so no model sees a day-ahead price that had not been auctioned
at its origin. Numbers produced here therefore SUPERSEDE everything in
`results/compare_rolling_dadrop_*.csv`, which was trained with that leak.

The design is a 2 x 3 grid plus references:

                    | market-rule block  | plain Dense mixer
    ----------------+--------------------+-------------------
    Dense encoder   | MRINN              | MLP
    LSTM encoder    | MR-LSTM            | LSTM
    GRU encoder     | MR-GRU             | GRU

    also: AttnBiLSTM (paper's strongest recurrent baseline), LQR (linear quantile
    regression, no hierarchical head), XGBoost (one multi-quantile booster per
    horizon), and the free references climatology + seasonal naive from baselines.py.

Reading DOWN a column isolates the encoder; reading ACROSS a row isolates the market
rules. The second comparison has never been run in this project.

Resumable: results append per (model, fold) and finished pairs are skipped, so an
interrupted run loses at most one fold.

    .venv/bin/python benchmark_models.py            # all nine
    .venv/bin/python benchmark_models.py gru mlp    # a subset
"""

import json
import os
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

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg",
                "P_aFRR_pos_MOL", "P_aFRR_neg_MOL", "P_ID15_nemo",
                "P_ID60_nemo", "P_DA_nemo"]
FEATS_CAP = ["L_ID15", "L_ID60", "L_DA"]
FEATS_VOL = ["system_imbalance", "E_aFRR_pos", "E_mFRR_pos",
             "E_aFRR_neg", "E_mFRR_neg"]
FEATS = FEATS_PRICES + FEATS_CAP + FEATS_VOL
LABEL = ["imbalance_price"]

# ------------------------------------------------------------------ recipe
T = 32
LAGS = list(range(1, T + 1))
HZ = MH.STEPS_PER_DAY
Q = list(MH.QUANTILES)
SEED = 42
STRIDE = 2
HU = 8
BATCH = 256              # 69 updates/epoch, not 18 -- PROJECT_REVIEW.md sec.4.2
BASE_LR = float(os.environ.get("BM_LR", 3e-3))   # with ReduceLROnPlateau, per the sweep
MAX_EPOCHS = int(os.environ.get("BM_EPOCHS", 300))   # ceiling; early stopping decides
PATIENCE = int(os.environ.get("BM_PATIENCE", 25))
DA_DROP = 0.2
GATE_CLOSURE = True      # the whole point of this rerun
# SMOKE=1 rehearses every model shape in the queue on one fold. The Keras-version trap
# that killed a 7-hour Kaggle run surfaced only at build time, on the one shape the
# rehearsal skipped -- so the rehearsal must build all of them.
SMOKE = os.environ.get("BM_SMOKE") == "1"
N_FOLDS = int(os.environ.get("BM_FOLDS", 0)) or None

TARGET_HORIZONS = [(1, "15 min"), (4, "1 hour"), (96, "1 day")]

OUT = Path("results")
RES_CSV = OUT / ("smoke_results.csv" if SMOKE else "benchmark_results.csv")
HOR_CSV = OUT / ("smoke_horizon.csv" if SMOKE else "benchmark_horizon.csv")

# name -> (family, builder kind). "mr" uses the rule blocks; "plain" does not.
#
# QUEUE ORDER IS DELIBERATE. Measured cost for 4 folds on this CPU runs from 4 min
# (LQR) to ~4 h (AttnBiLSTM), so the six models forming the 2 x 3 rules-x-encoder grid
# run first and the expensive extras last. If the run is cut short, what survives is
# still a complete, interpretable table rather than an arbitrary half of one.
MODELS = {
    "mrinn":      ("mr",    "mrinn"),      # grid: rules  x Dense
    "mlp":        ("plain", "mlp"),        # grid: plain  x Dense
    "lqr":        ("plain", "lqr"),        # cheap, and the only model without the head
    "mr_gru":     ("mr",    "gru"),        # grid: rules  x GRU
    "gru":        ("plain", "gru"),        # grid: plain  x GRU
    "mr_lstm":    ("mr",    "lstm"),       # grid: rules  x LSTM
    "lstm":       ("plain", "lstm"),       # grid: plain  x LSTM
    "xgb":        ("xgb",   "xgb"),
    "attnbilstm": ("plain", "attnbilstm"),
}
LABELS = {"mrinn": "MRINN", "mr_lstm": "MR-LSTM", "mr_gru": "MR-GRU",
          "mlp": "MLP", "lstm": "LSTM", "gru": "GRU",
          "attnbilstm": "AttnBiLSTM", "lqr": "LQR", "xgb": "XGBoost"}


def seed_all(seed: int) -> None:
    """Seed without `enable_op_determinism()`.

    `MR.set_random_seed` calls it, which disables the cuDNN RNN kernel and erases most
    of the GPU speedup. On CPU it costs less, but keeping the two paths identical
    matters more than the marginal determinism.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def input_kind(family: str, kind: str) -> str:
    """Which history layout `build_dataset` should emit: flat (N,T) or (N,T,1)."""
    if family == "mr":
        return "mrinn" if kind == "mrinn" else kind
    if kind in ("mlp", "lqr"):
        return "mrinn"          # flat (N, T)
    return "gru"                # (N, T, 1); cell type is irrelevant to the layout


def done_pairs() -> set:
    if not RES_CSV.exists():
        return set()
    d = pd.read_csv(RES_CSV)
    return set(zip(d.model, d.fold))


def append(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # A resumed file from an older schema misaligns a bare append; rewrite instead.
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def append_horizons(name: str, fold: str, per_h: pd.DataFrame) -> None:
    per_h = per_h.copy()
    per_h.insert(0, "fold", fold)
    per_h.insert(0, "model", name)
    if HOR_CSV.exists():
        per_h = pd.concat([pd.read_csv(HOR_CSV), per_h], ignore_index=True)
    per_h.to_csv(HOR_CSV, index=False)


# ------------------------------------------------------------------- training

def run_neural(name, family, kind, C, TRd, ytr, VAd, yva, TEd):
    seed_all(SEED)
    if family == "mr":
        model = MH.build_multihorizon(kind, T, C, hidden_units=HU, horizons=HZ,
                                      quantiles=Q, lr=BASE_LR, da_dropout=DA_DROP)
    else:
        model = B.build_baseline(kind, T, hidden_units=HU, horizons=HZ,
                                 quantiles=Q, lr=BASE_LR, da_dropout=DA_DROP)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=10, min_lr=1e-5, verbose=0),
    ]
    hist = model.fit(TRd, ytr, validation_data=(VAd, yva), epochs=MAX_EPOCHS,
                     batch_size=BATCH, callbacks=cbs, verbose=0)
    best = int(np.argmin(hist.history["val_loss"])) + 1
    return (model.predict(TEd, batch_size=1024, verbose=0), model.count_params(),
            best, len(hist.history["val_loss"]))


def run_xgb(TRd, ytr, VAd, yva, TEd):
    """One multi-quantile booster per horizon: XGBoost has no horizon axis."""
    preds = np.empty((len(TEd[0]), HZ, len(Q)), dtype="float32")
    n_trees = 0
    for h in range(1, HZ + 1):
        Xtr = B.xgb_features(TRd, h)
        Xva = B.xgb_features(VAd, h)
        Xte = B.xgb_features(TEd, h)
        m = B.fit_xgb_horizon(Xtr, ytr[:, h - 1], Xva, yva[:, h - 1], Q, seed=SEED)
        p = np.asarray(m.predict(Xte))
        preds[:, h - 1, :] = p.reshape(len(Xte), len(Q))
        n_trees += int(m.best_iteration or 0) + 1
    return preds, n_trees, 0, 0


# ----------------------------------------------------------------------- main

def main() -> None:
    wanted = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    OUT.mkdir(exist_ok=True)
    skip = done_pairs()

    print(f"models   {', '.join(wanted)}")
    print(f"recipe   T={T} stride={STRIDE} batch={BATCH} lr={BASE_LR} "
          f"epochs<={MAX_EPOCHS} patience={PATIENCE} da_drop={DA_DROP} seed={SEED}")
    print(f"gate_closure={GATE_CLOSURE}  <- these numbers supersede the leaky runs\n")

    data, _ = MR.load_data(FEATS_PRICES, FEATS_CAP, FEATS_VOL, LABEL,
                           str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv"))
    folds = MH.rolling_folds()[:N_FOLDS] if N_FOLDS else MH.rolling_folds()
    t_all = time.time()

    for name in wanted:
        family, kind = MODELS[name]
        for f in folds:
            if (name, f["tag"]) in skip:
                print(f"  [skip] {name} {f['tag']} -- already in {RES_CSV.name}")
                continue
            t0 = time.time()

            tr, va, te = MR.split_data(data, f["train"], f["val"], f["test"])
            trs, vas, tes, names = MR.shift_data(tr, va, te, LABEL[0], FEATS, [],
                                                 LAGS, [])
            Xtr, Xva, Xte, _, _, _, ysc = MR.scale_data(trs, vas, tes, names, LABEL)
            C = list(MR.scaled_params(tr, FEATS_PRICES, FEATS_CAP)[:11])

            ik = input_kind(family, kind)
            TRd, ytr, _ = MH.build_dataset(tr, trs, Xtr, LAGS, ik, ysc, stride=STRIDE,
                                           gate_closure=GATE_CLOSURE)
            VAd, yva, _ = MH.build_dataset(va, vas, Xva, LAGS, ik, ysc,
                                           gate_closure=GATE_CLOSURE)
            TEd, yte, stamps = MH.build_dataset(te, tes, Xte, LAGS, ik, ysc,
                                                gate_closure=GATE_CLOSURE)

            if family == "xgb":
                pred, params, best, epochs = run_xgb(TRd, ytr, VAd, yva, TEd)
            else:
                pred, params, best, epochs = run_neural(name, family, kind, C,
                                                        TRd, ytr, VAd, yva, TEd)

            # Invert to EUR/MWh inside the fold: each fold has its own scaler.
            yt = MH._inv(yte, ysc)
            yp = np.stack([MH._inv(pred[:, :, j], ysc) for j in range(len(Q))], axis=-1)
            per_h = MH.evaluate_by_horizon(yt, yp, Q, None)
            s = MH.summarise(per_h)
            dt = time.time() - t0

            append(RES_CSV, {"model": name, "label": LABELS[name], "family": family,
                             "fold": f["tag"], "seed": SEED, "T": T,
                             "gate_closure": GATE_CLOSURE, "params": params,
                             "best_epoch": best, "epochs": epochs, "rows": len(yt),
                             **{k: s[k] for k in ("AQL", "MAE", "RMSE", "AQCR",
                                                  "coverage80", "AIW")},
                             "train_sec": round(dt, 1)})
            append_horizons(name, f["tag"], per_h)
            if not SMOKE:
                np.savez_compressed(OUT / f"pred_{name}_{f['tag']}.npz",
                                    y_true=yt.astype("float32"),
                                    y_pred=yp.astype("float32"),
                                    stamps=stamps.astype("datetime64[ns]"))

            hit = " <-- hit the epoch ceiling" if epochs and epochs >= MAX_EPOCHS else ""
            print(f"  {LABELS[name]:<11s} {f['tag']}  AQL {s['AQL']:7.3f}  "
                  f"MAE {s['MAE']:6.2f}  AQCR {s['AQCR']:5.2f}  p {params:>7,}  "
                  f"ep {best}/{epochs}  {dt/60:5.1f} min{hit}")

            tf.keras.backend.clear_session()

    print(f"\ntotal {(time.time() - t_all)/3600:.2f} h")


if __name__ == "__main__":
    main()
