"""Train MRINN, MR-LSTM and MR-GRU on identical splits and compare.

Everything except the encoder is held constant: same splits, same scaling
(MRINN's per-column scale_data, so the MRINN rows are exactly as published),
same quantiles, same batch size, same optimiser, same epoch budget, same
checkpoint-on-best-val-loss policy, same seed. The only variable is the
encoder stage.

Usage:
    .venv/bin/python compare.py --epochs 50 --input-length 8
    .venv/bin/python compare.py --epochs 1            # timing calibration
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# data.make_shifts inserts lag columns one at a time; noisy but harmless.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import library_mrrnn as MR  # noqa: E402
from library_imbalance.model import build_MRINN, make_inference as mrinn_inference  # noqa: E402

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg",
                "P_aFRR_pos_MOL", "P_aFRR_neg_MOL",
                "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo"]
FEATS_CAPACITIES = ["L_ID15", "L_ID60", "L_DA"]
FEATS_VOLUME = ["system_imbalance", "E_aFRR_pos", "E_mFRR_pos",
                "E_aFRR_neg", "E_mFRR_neg"]
LABEL = ["imbalance_price"]

TRAIN_RANGE = ("2022-01-01 00:00:00+00:00", "2025-05-01 00:00:00+00:00")
VAL_RANGE   = ("2025-05-01 00:00:00+00:00", "2025-09-01 00:00:00+00:00")
TEST_RANGE  = ("2025-09-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00")

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
METRIC_KEYS = ["AQL", "AQCR", "AQCE", "AIW", "MAE", "RMSE", "R2"]


def prepare(regelzonen_data, feats, lags):
    df_train, df_val, df_test = MR.split_data(
        regelzonen_data, TRAIN_RANGE, VAL_RANGE, TEST_RANGE)
    tr_s, va_s, te_s, names = MR.shift_data(
        df_train, df_val, df_test, LABEL[0], feats, [], lags, [])
    split = MR.scale_data(tr_s, va_s, te_s, names, LABEL)
    return df_train, te_s, split


def run(args):
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)

    MR.set_random_seed(42)
    feats = FEATS_PRICES + FEATS_CAPACITIES + FEATS_VOLUME
    data_path = str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv")
    regelzonen_data, _ = MR.load_data(
        FEATS_PRICES, FEATS_CAPACITIES, FEATS_VOLUME, LABEL, data_path)

    T = args.input_length
    rows = {}

    # ---------------- naive baseline ----------------
    _, te_s1, split1 = prepare(regelzonen_data, feats, [1])
    y_scaler1 = split1[6]
    naive_res, _, _ = MR.naive_baseline(te_s1, LABEL[0], QUANTILES, y_scaler=y_scaler1)
    rows["Naive (persistence)"] = {**{k: naive_res[k] for k in METRIC_KEYS},
                                   "Params": 0, "TrainSec": 0.0, "T": 1}
    print(f"\n[naive] AQL={naive_res['AQL']:.2f} MAE={naive_res['MAE']:.2f}\n")

    configs = []
    # MRINN at its published config (single timestep)
    configs.append(("MRINN (T=1)", "mrinn", [1]))
    # MRINN given the same information as the recurrent models
    if T > 1:
        configs.append((f"MRINN (T={T})", "mrinn", list(range(1, T + 1))))
    configs.append((f"MR-LSTM (T={T})", "lstm", list(range(1, T + 1))))
    configs.append((f"MR-GRU (T={T})", "gru", list(range(1, T + 1))))

    for label, kind, lags in configs:
        print("=" * 70)
        print(f"{label}  |  lags={lags}  epochs={args.epochs}")
        print("=" * 70)

        df_train, te_s, split = prepare(regelzonen_data, feats, lags)
        X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = split
        print(f"rows: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

        Cs = MR.scaled_params(df_train, FEATS_PRICES, FEATS_CAPACITIES)
        C_kwargs = {f"C{i}": Cs[i] for i in range(11)}

        MR.set_random_seed(42)
        ckpt = str(outdir / f"best_{kind}_T{len(lags)}.keras")
        t0 = time.perf_counter()

        if kind == "mrinn":
            # build_MRINN hardcodes its checkpoint filename; run from outdir
            # so concurrent configs do not clobber each other.
            import os
            cwd = os.getcwd()
            os.chdir(outdir)
            try:
                model, _ = build_MRINN(
                    "ImbalancePrice", X_train, y_train, X_val, y_val,
                    hidden_units=args.hidden, num_layer=args.mrinn_layers,
                    epochs=args.epochs, **C_kwargs,
                    lags=lags, quantiles=QUANTILES)
                train_sec = time.perf_counter() - t0
                yqs = mrinn_inference(X_test, lags, QUANTILES)
            finally:
                os.chdir(cwd)
        else:
            model, _ = MR.build_MRRNN(
                "ImbalancePrice", X_train, y_train, X_val, y_val,
                hidden_units=args.hidden, num_layer=1, epochs=args.epochs,
                **C_kwargs, cell=kind, lags=lags, quantiles=QUANTILES,
                checkpoint_path=ckpt)
            train_sec = time.perf_counter() - t0
            yqs = MR.make_inference(X_test, lags, QUANTILES, checkpoint_path=ckpt)

        res = MR.evaluate_performance(y_test, yqs, QUANTILES, y_scaler, verbose=False)
        rows[label] = {**{k: res[k] for k in METRIC_KEYS},
                       "Params": model.count_params(),
                       "TrainSec": round(train_sec, 1),
                       "T": len(lags)}
        print(f"\n-> AQL={res['AQL']:.3f}  AQCR={res['AQCR']:.2f}  "
              f"MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}  "
              f"params={model.count_params()}  train={train_sec:.0f}s\n")

        # checkpoint results after every config, so a long run is never lost
        (outdir / f"comparison_e{args.epochs}_T{T}.json").write_text(
            json.dumps(rows, indent=2))

    table = pd.DataFrame(rows).T[["T", "AQL", "AQCR", "AQCE", "AIW",
                                  "MAE", "RMSE", "Params", "TrainSec"]]
    print("\n" + "=" * 70)
    print(f"COMPARISON  (epochs={args.epochs}, hidden={args.hidden}, "
          f"quantiles={QUANTILES})")
    print("=" * 70)
    print(table.to_string())
    print(
        "\nNOTE: do NOT compare these to MRINN's published Table 3 row"
        "\n(AQL 20.70, MAE 49.36, RMSE 277.33). That row was computed on a"
        "\ndifferent test split. This repo's tutorial test window"
        "\n(2025-09-01..2026-01-01) is unusually calm: price std 83 and 6"
        "\nspikes above 1000 EUR/MWh, against std 387 and a +/-15000 range"
        "\nover the full CSV. Absolute errors here are therefore much lower."
        "\nThe valid comparison is the MRINN row above vs the MR-* rows,"
        "\nwhich were trained and scored under identical conditions."
    )

    table.to_csv(outdir / f"comparison_e{args.epochs}_T{T}.csv")
    print(f"\nsaved -> {outdir}/comparison_e{args.epochs}_T{T}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--input-length", type=int, default=8)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--mrinn-layers", type=int, default=2)
    run(p.parse_args())
