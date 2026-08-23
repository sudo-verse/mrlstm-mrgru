"""End-to-end verification for MR-LSTM / MR-GRU.

Runs the checks from the implementation plan on a small, fast configuration
before any long training run:

  1. the MRINN checkout is unmodified
  2. input arrays line up with the model's inputs
  3. the time axis runs oldest -> newest
  4. the parameter budget is where the analysis said it would be
  5. a short end-to-end fit + checkpoint reload works
  6. AQCR == 0 (the hierarchical head is doing its job)

Usage:  .venv/bin/python verify.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import library_mrrnn as MR  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = MR.MRINN_ROOT / "Data" / "imbalance_data.csv"

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg",
                "P_aFRR_pos_MOL", "P_aFRR_neg_MOL",
                "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo"]
FEATS_CAPACITIES = ["L_ID15", "L_ID60", "L_DA"]
FEATS_VOLUME = ["system_imbalance", "E_aFRR_pos", "E_mFRR_pos",
                "E_aFRR_neg", "E_mFRR_neg"]
LABEL = ["imbalance_price"]

T = 4
LAGS = list(range(1, T + 1))
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
HIDDEN = 8

_failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failures.append(name)
    return condition


def main():
    print("\n=== 1. MRINN checkout is unmodified ===")
    # Only tracked-file edits matter. Importing library_imbalance drops
    # .pyc files into its __pycache__ (upstream even commits some), and
    # untracked bytecode is not a modification of the market rules.
    try:
        edits = []
        for args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"]):
            out = subprocess.run(
                ["git", "-C", str(MR.MRINN_ROOT)] + args,
                capture_output=True, text=True, timeout=30,
            )
            edits += [ln for ln in out.stdout.split() if ln]
        check("vendored MRINN sources unmodified", not edits,
              ", ".join(sorted(set(edits)))[:200] or "clean")
    except Exception as exc:  # pragma: no cover
        check("git status ran", False, str(exc))

    print("\n=== 2. Data pipeline ===")
    MR.set_random_seed(42)
    feats = FEATS_PRICES + FEATS_CAPACITIES + FEATS_VOLUME
    regelzonen_data, _ = MR.load_data(
        FEATS_PRICES, FEATS_CAPACITIES, FEATS_VOLUME, LABEL, str(DATA))

    # short ranges -- this is a smoke test, not a training run
    df_train, df_val, df_test = MR.split_data(
        regelzonen_data,
        ("2022-01-01 00:00:00+00:00", "2022-03-01 00:00:00+00:00"),
        ("2022-03-01 00:00:00+00:00", "2022-04-01 00:00:00+00:00"),
        ("2022-04-01 00:00:00+00:00", "2022-05-01 00:00:00+00:00"),
    )
    tr_s, va_s, te_s, feature_names = MR.shift_data(
        df_train, df_val, df_test, LABEL[0], feats, [], LAGS, [])
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = \
        MR.scale_data_shared_lags(tr_s, va_s, te_s, feature_names, LABEL)
    check("splits are non-empty", min(len(X_train), len(X_val), len(X_test)) > 0,
          f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    print("\n=== 3. Time axis runs oldest -> newest ===")
    seqs = MR.to_sequence_inputs(X_test, lags=LAGS)
    base0 = MR.FEATURE_BASES[0]
    s0 = seqs[0]
    check("shape is (N, T, 1)", s0.shape == (len(X_test), T, 1), str(s0.shape))
    check(f"last step == {base0}_lag1 (most recent)",
          np.allclose(s0[:, -1, 0], X_test[f"{base0}_lag1"].to_numpy(np.float32)))
    check(f"first step == {base0}_lag{T} (oldest)",
          np.allclose(s0[:, 0, 0], X_test[f"{base0}_lag{T}"].to_numpy(np.float32)))

    print("\n=== 4. Constants scaled against the training split ===")
    Cs = MR.scaled_params(df_train, FEATS_PRICES, FEATS_CAPACITIES)
    C_kwargs = {f"C{i}": Cs[i] for i in range(11)}
    check("C0..C10 all finite", all(np.isfinite(c) for c in Cs[:11]),
          f"C0={Cs[0]:.4f} C4={Cs[4]:.4f} C10={Cs[10]:.4f}")

    # Expected parameter counts, computed rather than hardcoded.
    #
    # Only 16 of the 17 encoders survive: get_P_EX never consumes L_DA_rep
    # (it derives w_DA = 1 - w_ID15 - w_ID60), so Keras prunes that branch.
    # L_DA is doubly dead -- data.load_data also pins the column to 0.
    # The L_DA input still exists on the model, so it must still be fed.
    H, N_ENC = HIDDEN, 16
    gate_and_head = 391 - 2 * (H + 1) * (5 - len(QUANTILES))
    expected_params = {
        "gru": N_ENC * 3 * (1 * H + H * H + 2 * H) + gate_and_head,
        "lstm": N_ENC * 4 * (1 * H + H * H + H) + gate_and_head,
    }

    tmpdir = Path(tempfile.mkdtemp(prefix="mrrnn_verify_"))
    try:
        for cell, expected in ((c, expected_params[c]) for c in ("gru", "lstm")):
            print(f"\n=== 5. Build + fit: MR-{cell.upper()} ===")
            ckpt = tmpdir / f"best_{cell}.keras"
            model, history = MR.build_MRRNN(
                "ImbalancePrice", X_train, y_train, X_val, y_val,
                hidden_units=HIDDEN, num_layer=1, epochs=2,
                **C_kwargs, cell=cell, lags=LAGS, quantiles=QUANTILES,
                checkpoint_path=str(ckpt), verbose=0,
            )
            n = model.count_params()
            check(f"MR-{cell.upper()} params == projected {expected}",
                  n == expected, f"{n} vs {expected}")
            check("params below MLP baseline (8.9k)", n < 8900, str(n))
            check("input alignment guard passed", True, "asserted during build")
            check("loss is finite", np.isfinite(history.history["loss"][-1]),
                  f"{history.history['loss'][-1]:.4f}")
            check("checkpoint written", ckpt.exists())

            print(f"\n=== 6. Reload + inference: MR-{cell.upper()} ===")
            yqs = MR.make_inference(X_test, LAGS, QUANTILES, checkpoint_path=str(ckpt))
            check("one array per quantile", len(yqs) == len(QUANTILES))

            results = MR.evaluate_performance(
                y_test, yqs, QUANTILES, y_scaler, verbose=False)
            check("AQCR == 0 (no quantile crossing)", results["AQCR"] == 0.0,
                  f"{results['AQCR']}")
            check("MAE finite", np.isfinite(results["MAE"]),
                  f"MAE={results['MAE']:.2f} AQL={results['AQL']:.2f} "
                  f"RMSE={results['RMSE']:.2f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
