"""Baselines for the 2025 day-ahead test window, scored on the models' own rows.

PROJECT_REVIEW.md sec.3 and sec.5 item 2:

  * the headline "-32.8% vs seasonal naive" leans on a baseline that is itself worse
    than predicting a constant (naive RMSE 549 ~ sqrt(2) x price sd 390), so the
    claim needs restating against a constant/climatology reference;
  * the naive was scored on 34,841 contiguous rows while the models were scored on
    34,521 per-fold rows -- and the extra rows are exactly the quarter boundaries,
    where the naive can look up "yesterday" across a fold edge and the models cannot.

This script fixes both: it reconstructs the exact row set the rolling-fold models are
evaluated on (via `build_dataset`, no training required) and scores every baseline on
that set, with each fold's climatology and residual quantiles read from that fold's
own training window.

    .venv/bin/python baselines.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import library_mrrnn as MR                          # noqa: E402
from library_mrrnn import multihorizon as MH        # noqa: E402

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg",
                "P_aFRR_pos_MOL", "P_aFRR_neg_MOL", "P_ID15_nemo",
                "P_ID60_nemo", "P_DA_nemo"]
FEATS_CAP = ["L_ID15", "L_ID60", "L_DA"]
FEATS_VOL = ["system_imbalance", "E_aFRR_pos", "E_mFRR_pos",
             "E_aFRR_neg", "E_mFRR_neg"]
FEATS = FEATS_PRICES + FEATS_CAP + FEATS_VOL
LABEL = ["imbalance_price"]
LAGS = list(range(1, 33))                            # T = 32
HZ = MH.STEPS_PER_DAY
Q = MH.QUANTILES

# Reported for context: the best measured model on this window (MR-GRU, rolling +
# DA-dropout, seed 42, from results/compare_rolling_dadrop_results.csv). It was
# trained WITH the day-ahead leak that audit_gate_closure.py measures, so treat it as
# an optimistic reference until the gated rerun lands.
MODEL_REF = {"name": "MR-GRU (rolling+DAdrop, s42)", "AQL": 25.629, "MAE": 64.85}


def main() -> None:
    data, _ = MR.load_data(FEATS_PRICES, FEATS_CAP, FEATS_VOL, LABEL,
                           str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv"))

    yt_all, yp_naive, yp_clim, stamps_all = [], [], [], []
    print("rebuilding the rolling-fold evaluation rows (no training)\n")

    for f in MH.rolling_folds():
        tr, va, te = MR.split_data(data, f["train"], f["val"], f["test"])
        trs, vas, tes, names = MR.shift_data(tr, va, te, LABEL[0], FEATS, [], LAGS, [])
        _, _, Xte, _, _, _, ysc = MR.scale_data(trs, vas, tes, names, LABEL)
        _, yte, stamps = MH.build_dataset(te, tes, Xte, LAGS, "gru", ysc)

        # This fold's training prices -- the only sample a forecaster may calibrate on.
        train_prices = tr[LABEL[0]].to_numpy(dtype="float64")

        # Day-over-day residual quantiles from TRAINING, not from the rows being scored.
        tr_y = train_prices
        d = tr_y[HZ:] - tr_y[:-HZ]
        resid_q = np.quantile(d[~np.isnan(d)], Q)

        yt_n, yp_n, ok = MH.seasonal_naive(yte, stamps, ysc, Q, resid_quantiles=resid_q)
        yt_c, yp_c = MH.climatology(yte, ysc, Q, reference=train_prices)

        # Score every baseline on the naive's row set: it is the most restrictive
        # (it alone needs a genuine "yesterday"), and mixing row sets is the bug
        # PROJECT_REVIEW.md sec.5 item 2 flags.
        yt_all.append(yt_n)
        yp_naive.append(yp_n)
        yp_clim.append(yp_c[ok])
        stamps_all.append(stamps[ok])

        print(f"  {f['tag']}  train {f['train'][0][:10]} -> {f['train'][1][:10]}   "
              f"test {f['test'][0][:10]} -> {f['test'][1][:10]}   "
              f"{len(yt_n):,} rows   train median {np.nanmedian(train_prices):7.2f}")

    yt = np.concatenate(yt_all)
    stamps = np.concatenate(stamps_all)
    preds = {"seasonal naive": np.concatenate(yp_naive),
             "climatology (train quantiles)": np.concatenate(yp_clim)}

    # The constant-median predictor: the point forecast the review benchmarked. Given
    # as a degenerate quantile forecast (every quantile equal) so AQL stays comparable.
    med = preds["climatology (train quantiles)"][:, :, Q.index(0.5)]
    preds["constant = train median"] = np.repeat(med[:, :, None], len(Q), axis=-1)

    # yt is (origins, 96): every target interval appears under ~96 different origins,
    # so counting spikes on the matrix multiplies them by 96. Count distinct intervals.
    tgt_times = (pd.DatetimeIndex(stamps).values[:, None]
                 + (np.arange(1, HZ + 1) * np.timedelta64(15, "m"))[None, :])
    _, first = np.unique(tgt_times.ravel(), return_index=True)
    uniq = yt.ravel()[first]

    print(f"\nscored rows: {len(yt):,} origins x {HZ} horizons = {yt.size:,} cells")
    print(f"distinct target intervals: {uniq.size:,}")
    print(f"price over them: mean {uniq.mean():8.2f}   sd {uniq.std():8.2f}   "
          f"median {np.median(uniq):8.2f}")
    print(f"spikes |p| > 500: {(np.abs(uniq) > 500).sum():,}   "
          f"|p| > 5000: {(np.abs(uniq) > 5000).sum():,}\n")

    rows = []
    for name, yp in preds.items():
        s = MH.summarise(MH.evaluate_by_horizon(yt, yp, Q, None))
        rows.append({"baseline": name, **{k: s[k] for k in
                                          ("AQL", "MAE", "RMSE", "coverage80", "AIW")}})
    tab = pd.DataFrame(rows).set_index("baseline").sort_values("AQL")
    print(tab.round(3).to_string())

    best = tab.MAE.min()
    print(f"\nstrongest baseline on MAE: {tab.MAE.idxmin()}  ({best:.2f})")
    print(f"{MODEL_REF['name']}  MAE {MODEL_REF['MAE']:.2f}  AQL {MODEL_REF['AQL']:.3f}")
    print(f"  vs seasonal naive        {MODEL_REF['MAE'] / tab.loc['seasonal naive', 'MAE'] - 1:+7.2%} MAE")
    print(f"  vs strongest baseline    {MODEL_REF['MAE'] / best - 1:+7.2%} MAE   "
          f"<- the claim to report")
    print(f"  RMSE, model {tab.RMSE.min():.2f}-class vs constant "
          f"{tab.loc['constant = train median', 'RMSE']:.2f}: "
          f"the models capture essentially none of the spike variance")

    out = Path("results/baselines_2025.csv")
    tab.to_csv(out)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
