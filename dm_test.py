"""Diebold-Mariano tests on the pooled multi-horizon forecasts.

PROJECT_REVIEW.md open issue 4: a DM test exists in `experiments.py` but is wired only
to the single-step path, and porting it needs care because

    adjacent origins share 95 of their 96 targets, so the standard HAC lag badly
    understates the variance -- use lag >= 96 or score non-overlapping daily origins.

Both remedies are implemented here and reported side by side:

  * NEWEY-WEST over every origin with a lag of at least 96, so the autocovariance of the
    overlap is actually inside the bandwidth; and
  * NON-OVERLAPPING, which keeps one origin per day so successive loss differentials
    share no targets at all and the classical DM assumption holds outright.

If the two disagree, believe the non-overlapping one -- it buys its honesty with a
smaller sample rather than with a variance correction.

The loss differential is the per-origin mean pinball loss over all 96 horizons and all
five quantiles, which is the AQL the tables report.

    .venv/bin/python dm_test.py
    .venv/bin/python dm_test.py --h 1        # test at a single horizon
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results")
QUANTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
STEPS_PER_DAY = 96

PAIRS = [
    ("mr_gru", "mrinn",   "encoder: GRU vs Dense, rules held"),
    ("mr_lstm", "mrinn",  "encoder: LSTM vs Dense, rules held"),
    ("mr_gru", "mr_lstm", "encoder: GRU vs LSTM, rules held"),
    ("mr_gru", "gru",     "rules: on vs off, GRU encoder"),
    ("mr_lstm", "lstm",   "rules: on vs off, LSTM encoder"),
    ("mrinn", "mlp",      "rules: on vs off, Dense encoder"),
    ("mr_gru", "attnbilstm", "MR-GRU vs the paper's strongest recurrent baseline"),
    ("mr_gru", "xgb",     "MR-GRU vs gradient boosting"),
]


def load(model: str):
    files = sorted(OUT.glob(f"pred_{model}_*.npz"))
    if not files:
        return None, None, None
    yt = np.concatenate([np.load(f)["y_true"] for f in files])
    yp = np.concatenate([np.load(f)["y_pred"] for f in files])
    st = np.concatenate([np.load(f)["stamps"] for f in files])
    return yt, yp, st


def pinball_per_origin(yt, yp, horizon=None):
    """Mean pinball loss per origin. `horizon` restricts to one h (1-indexed)."""
    if horizon is not None:
        yt = yt[:, horizon - 1:horizon]
        yp = yp[:, horizon - 1:horizon, :]
    e = yt[:, :, None] - yp
    return np.mean(np.maximum(QUANTILES * e, (QUANTILES - 1.0) * e), axis=(1, 2))


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of the mean of `d` with a Bartlett kernel."""
    n = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    total = gamma0
    for k in range(1, lag + 1):
        cov = np.dot(dc[k:], dc[:-k]) / n
        total += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return max(total, 1e-18) / n


def dm(d: np.ndarray, lag: int):
    var = newey_west_var(d, lag)
    stat = d.mean() / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p


def dm_nonoverlap(d: np.ndarray, stamps: np.ndarray):
    """One origin per day: successive differentials then share no targets at all."""
    idx = pd.DatetimeIndex(stamps)
    keep = np.zeros(len(idx), dtype=bool)
    seen = set()
    for i, day in enumerate(idx.normalize()):
        if day not in seen:
            seen.add(day)
            keep[i] = True
    dd = d[keep]
    stat = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    p = 2 * (1 - stats.t.cdf(abs(stat), df=len(dd) - 1))
    return stat, p, len(dd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=None, help="restrict to one horizon")
    ap.add_argument("--lag", type=int, default=STEPS_PER_DAY,
                    help="Newey-West lag (default 96 = the overlap length)")
    args = ap.parse_args()

    cache = {}
    scope = f"horizon h={args.h}" if args.h else "all 96 horizons"
    print(f"Diebold-Mariano on pooled rolling-fold forecasts -- {scope}")
    print(f"loss = mean pinball over 5 quantiles; Newey-West lag {args.lag}; "
          f"negative statistic favours the FIRST model\n")

    rows = []
    for a, b, why in PAIRS:
        for m in (a, b):
            if m not in cache:
                cache[m] = load(m)
        (yta, ypa, sta), (ytb, ypb, _) = cache[a], cache[b]
        if yta is None or ytb is None:
            continue
        if not np.array_equal(yta, ytb):
            print(f"  !! {a} vs {b}: targets differ -- models were scored on "
                  f"different rows, comparison skipped")
            continue
        da = pinball_per_origin(yta, ypa, args.h)
        db = pinball_per_origin(ytb, ypb, args.h)
        d = da - db

        stat, p = dm(d, args.lag)
        nstat, np_, nn = dm_nonoverlap(d, sta)
        rows.append({"comparison": f"{a} - {b}", "meaning": why,
                     "mean_dAQL": float(d.mean()), "DM_NW": stat, "p_NW": p,
                     "DM_daily": nstat, "p_daily": np_, "n_daily": nn})

    if not rows:
        print("no completed model pairs yet")
        return

    t = pd.DataFrame(rows)
    show = t[["comparison", "mean_dAQL", "DM_NW", "p_NW", "DM_daily", "p_daily"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\n  n = {len(cache[PAIRS[0][0]][0]):,} origins; "
          f"non-overlapping uses {rows[0]['n_daily']:,} (one per day)")
    print("\n  interpretation")
    for r in rows:
        verdict = ("significant at 5% on BOTH tests" if r["p_NW"] < .05 and r["p_daily"] < .05
                   else "significant on the Newey-West test only -- treat as suggestive"
                   if r["p_NW"] < .05
                   else "not significant")
        winner = r["comparison"].split(" - ")[0 if r["mean_dAQL"] < 0 else 1]
        print(f"    {r['meaning']:<52s} -> {winner:>10s}, {verdict}")

    t.to_csv(OUT / "dm_tests.csv", index=False)
    print(f"\nwritten: {OUT / 'dm_tests.csv'}")


if __name__ == "__main__":
    main()
