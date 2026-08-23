"""Phase A (seed replication) and Phase B (input-length sweep).

Phase A answers "is the MR-* vs MRINN gap real, or one lucky seed?" by
repeating the T=8 comparison over several seeds and reporting mean +/- sd
plus a Diebold-Mariano test on per-observation loss differentials.

Phase B answers "does more temporal context help?" by sweeping the input
length. T=1 doubles as a pipeline control -- it should land near MRINN.

Every finished run is appended to results/runs.csv immediately and the
per-observation pinball losses are saved, so a crash never loses completed
work and the script resumes where it stopped.

    .venv/bin/python experiments.py --phase A
    .venv/bin/python experiments.py --phase B
    .venv/bin/python experiments.py --phase both
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import library_mrrnn as MR  # noqa: E402
from library_imbalance.model import build_MRINN, make_inference as mrinn_inference  # noqa: E402
from compare import (FEATS_CAPACITIES, FEATS_PRICES, FEATS_VOLUME, LABEL,  # noqa: E402
                     METRIC_KEYS, QUANTILES, prepare)

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
RUNS_CSV = OUT / "runs.csv"
LOSS_DIR = OUT / "losses"
LOSS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
PHASE_B_T = [1, 4, 16, 32]          # T=8 already covered by phase A
MODELS = ["mrinn", "lstm", "gru"]


# ---------------------------------------------------------------- utilities

def pinball_per_obs(y_true, yqs, quantiles, y_scaler):
    """Per-observation pinball loss, averaged over quantiles, original scale."""
    inv = lambda a: y_scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
    yt = inv(np.asarray(y_true).ravel())
    losses = []
    for q, yq in zip(quantiles, yqs):
        e = yt - inv(yq)
        losses.append(np.maximum(q * e, (q - 1.0) * e))
    return np.mean(np.stack(losses, axis=1), axis=1)      # (N,)


def diebold_mariano(loss_a, loss_b):
    """DM test on loss differentials with Newey-West HAC variance.

    Returns (stat, p_value). Negative stat favours model A.
    """
    from math import erfc, sqrt
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))     # standard lag rule
    gamma0 = float(dc @ dc) / n
    var = gamma0
    for lag in range(1, L + 1):
        g = float(dc[lag:] @ dc[:-lag]) / n
        var += 2.0 * (1.0 - lag / (L + 1.0)) * g
    if var <= 0:
        return float("nan"), float("nan")
    stat = dbar / sqrt(var / n)
    p = erfc(abs(stat) / sqrt(2.0))                        # two-sided normal
    return float(stat), float(p)


def load_runs():
    if RUNS_CSV.exists():
        return pd.read_csv(RUNS_CSV)
    return pd.DataFrame()


def already_done(runs, model, T, seed):
    if runs.empty:
        return False
    m = (runs["model"] == model) & (runs["T"] == T) & (runs["seed"] == seed)
    return bool(m.any())


def append_run(row):
    df = pd.DataFrame([row])
    df.to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)


# ---------------------------------------------------------------- one run

def run_one(regelzonen_data, feats, model, T, seed, epochs):
    lags = list(range(1, T + 1))
    tag = f"{model}_T{T}_s{seed}"
    print("=" * 72)
    print(f"{tag}   epochs={epochs}")
    print("=" * 72, flush=True)

    df_train, _, split = prepare(regelzonen_data, feats, lags)
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = split

    Cs = MR.scaled_params(df_train, FEATS_PRICES, FEATS_CAPACITIES)
    C_kwargs = {f"C{i}": Cs[i] for i in range(11)}

    MR.set_random_seed(seed)
    ckpt = str(OUT / f"ckpt_{tag}.keras")
    t0 = time.perf_counter()

    if model == "mrinn":
        cwd = os.getcwd()
        os.chdir(OUT)                      # build_MRINN hardcodes its filename
        try:
            m, _ = build_MRINN("ImbalancePrice", X_train, y_train, X_val, y_val,
                               hidden_units=8, num_layer=2, epochs=epochs,
                               **C_kwargs, lags=lags, quantiles=QUANTILES)
            train_sec = time.perf_counter() - t0
            yqs = mrinn_inference(X_test, lags, QUANTILES)
        finally:
            os.chdir(cwd)
    else:
        m, _ = MR.build_MRRNN("ImbalancePrice", X_train, y_train, X_val, y_val,
                              hidden_units=8, num_layer=1, epochs=epochs,
                              **C_kwargs, cell=model, lags=lags,
                              quantiles=QUANTILES, checkpoint_path=ckpt)
        train_sec = time.perf_counter() - t0
        yqs = MR.make_inference(X_test, lags, QUANTILES, checkpoint_path=ckpt)

    res = MR.evaluate_performance(y_test, yqs, QUANTILES, y_scaler, verbose=False)
    np.save(LOSS_DIR / f"{tag}.npy",
            pinball_per_obs(y_test, yqs, QUANTILES, y_scaler))

    row = {"model": model, "T": T, "seed": seed,
           **{k: res[k] for k in METRIC_KEYS},
           "Params": m.count_params(), "TrainSec": round(train_sec, 1),
           "n_test": len(X_test)}
    append_run(row)
    print(f"-> AQL={res['AQL']:.4f}  MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}  "
          f"AQCE={res['AQCE']:.3f}  params={m.count_params()}  {train_sec:.0f}s\n",
          flush=True)

    # free the graph between runs; 40+ models accumulate otherwise
    import tensorflow as tf
    tf.keras.backend.clear_session()
    return row


# ---------------------------------------------------------------- phases

def phase_a(regelzonen_data, feats, epochs):
    print("\n" + "#" * 72)
    print("# PHASE A - seed replication at T=8")
    print("#" * 72 + "\n", flush=True)
    runs = load_runs()
    plan = [("mrinn", 1), ("mrinn", 8), ("lstm", 8), ("gru", 8)]
    for seed in SEEDS:
        for model, T in plan:
            if already_done(runs, model, T, seed):
                print(f"skip {model}_T{T}_s{seed} (done)", flush=True)
                continue
            run_one(regelzonen_data, feats, model, T, seed, epochs)


def phase_b(regelzonen_data, feats, epochs):
    print("\n" + "#" * 72)
    print("# PHASE B - input-length sweep at seed 42")
    print("#" * 72 + "\n", flush=True)
    runs = load_runs()
    for T in PHASE_B_T:
        for model in MODELS:
            if T == 1 and model == "mrinn":
                continue                       # identical to phase A's mrinn_T1
            if already_done(runs, model, T, 42):
                print(f"skip {model}_T{T}_s42 (done)", flush=True)
                continue
            run_one(regelzonen_data, feats, model, T, 42, epochs)


# ---------------------------------------------------------------- reporting

def summarise():
    runs = load_runs()
    if runs.empty:
        print("no runs yet")
        return

    print("\n" + "=" * 72)
    print("PHASE A - mean +/- sd across seeds (T=8 unless noted)")
    print("=" * 72)
    a = runs[runs["seed"].isin(SEEDS) & (runs["T"].isin([1, 8]))]
    if not a.empty:
        g = a.groupby(["model", "T"])
        summary = pd.DataFrame({
            "n_seeds": g["seed"].nunique(),
            "AQL_mean": g["AQL"].mean().round(4),
            "AQL_sd": g["AQL"].std().round(4),
            "MAE_mean": g["MAE"].mean().round(3),
            "MAE_sd": g["MAE"].std().round(3),
            "RMSE_mean": g["RMSE"].mean().round(3),
            "AQCE_mean": g["AQCE"].mean().round(3),
            "Params": g["Params"].first(),
        })
        print(summary.to_string())

        # Diebold-Mariano vs the MRINN (T=1) reference, pooled over seeds
        print("\nDiebold-Mariano vs MRINN (T=1), per seed")
        print("-" * 72)
        print(f"{'model':<12}{'seed':>6}{'DM stat':>12}{'p-value':>12}  verdict")
        for model, T in [("mrinn", 8), ("lstm", 8), ("gru", 8)]:
            for seed in SEEDS:
                fa = LOSS_DIR / f"{model}_T{T}_s{seed}.npy"
                fb = LOSS_DIR / f"mrinn_T1_s{seed}.npy"
                if not (fa.exists() and fb.exists()):
                    continue
                la, lb = np.load(fa), np.load(fb)
                n = min(len(la), len(lb))
                stat, p = diebold_mariano(la[-n:], lb[-n:])
                verdict = ("better" if stat < 0 else "worse") if p < 0.05 else "no diff"
                print(f"{model+' T'+str(T):<12}{seed:>6}{stat:>12.3f}{p:>12.4f}  {verdict}")

    print("\n" + "=" * 72)
    print("PHASE B - input-length sweep (seed 42)")
    print("=" * 72)
    b = runs[runs["seed"] == 42].sort_values(["T", "model"])
    if not b.empty:
        print(b[["model", "T", "AQL", "AQCE", "MAE", "RMSE", "Params",
                 "TrainSec", "n_test"]].to_string(index=False))

    runs.to_csv(OUT / "runs.csv", index=False)
    print(f"\nall runs -> {RUNS_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["A", "B", "both", "report"], default="both")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    if args.phase != "report":
        MR.set_random_seed(42)
        feats = FEATS_PRICES + FEATS_CAPACITIES + FEATS_VOLUME
        data = MR.load_data(FEATS_PRICES, FEATS_CAPACITIES, FEATS_VOLUME,
                            LABEL, str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv"))[0]
        if args.phase in ("A", "both"):
            phase_a(data, feats, args.epochs)
        if args.phase in ("B", "both"):
            phase_b(data, feats, args.epochs)

    summarise()
