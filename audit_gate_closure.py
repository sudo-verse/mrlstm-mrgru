"""Day-ahead gate-closure audit -- is `make_da_future` leaking unpublished prices?

PROJECT_REVIEW.md sec.5 item 4 flags this as the one unresolved issue that could move
the headline numbers *unfavourably*:

    `make_da_future` takes shift(-h) for all 96 horizons unconditionally. DA prices for
    delivery day D are published ~12:00-13:00 CET on D-1, so early-morning origins use
    prices not yet published.

This script quantifies the exposure directly from `imbalance_data.csv`. It answers:

  1. What fraction of (origin, horizon) cells consume a price not yet published?
  2. Which origins and which horizons carry the leak?
  3. Does the leak actually carry information, or is it harmless because the leaked
     price is close to something the model could have known legitimately?
  4. What does a gate-closure-correct `present` flag look like, and how much of the
     day-ahead tensor does it blank?

Nothing here trains anything. Run it with the repo venv:

    .venv/bin/python audit_gate_closure.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration

DATA = Path("/home/sudoverse/fyp/imbalance_data.csv")
TIME_COL = "Time [UTC] start"
DA_COL = "P_DA_nemo"
PRICE_COL = "imbalance_price"
HORIZONS = 96
STEP = pd.Timedelta("15min")

# The single-price-coupled day-ahead auction closes 12:00 CET on D-1 and results are
# published shortly after (SDAC target is 12:55; the exchanges quote "around 13:00").
# `MARKET_TZ` is the delivery-day calendar, which is what the auction is defined over --
# both the Austrian and the German bidding zone use it, so the Austrian/German ambiguity
# in the source description does not change this audit.
MARKET_TZ = "Europe/Vienna"
PUBLISH_LOCAL_HOUR = 13  # conservative: assume results are usable from 13:00 local

TEST_START, TEST_END = "2025-01-01", "2026-01-01"


# ------------------------------------------------------------------- gate model

def publication_instant(stamps_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC instant at which each timestamp's day-ahead price became public.

    The delivery day is a *local* calendar day, so the publication instant is
    13:00 local on the preceding local day -- which is 12:00 UTC in winter and
    11:00 UTC in summer. Doing this in UTC arithmetic would be wrong by an hour
    for half the year, and the leak is an hours-scale effect.
    """
    # Arithmetic on tz-AWARE timestamps is absolute, so "midnight minus one day plus
    # 13 h" lands at 12:00 or 14:00 local on the two DST changeover days. Do the day
    # arithmetic on naive wall-clock and re-localize, which is what "13:00 local" means.
    local_naive = stamps_utc.tz_convert(MARKET_TZ).tz_localize(None)
    pub_naive = (local_naive.normalize() - pd.Timedelta(days=1)
                 + pd.Timedelta(hours=PUBLISH_LOCAL_HOUR))
    return (pd.DatetimeIndex(pub_naive)
            .tz_localize(MARKET_TZ, nonexistent="shift_forward", ambiguous=True)
            .tz_convert("UTC"))


def leak_matrix(stamps: pd.DatetimeIndex, horizons: int = HORIZONS) -> np.ndarray:
    """(N, horizons) bool -- True where origin i consumes an unpublished price at h.

    Cell (i, h) is the day-ahead price for target time stamps[i + h]. It is legitimate
    only if that price was already public at the origin instant stamps[i].
    """
    pub = publication_instant(stamps).to_numpy()
    origin = stamps.to_numpy()
    n = len(stamps)
    out = np.zeros((n, horizons), dtype=bool)
    for h in range(1, horizons + 1):
        tgt = np.arange(n) + h
        valid = tgt < n
        out[valid, h - 1] = pub[tgt[valid]] > origin[valid]
    # Rows whose full 96-step target vector runs off the end are dropped by
    # `build_dataset`'s NaN cut, so they never reach a model. Exclude them here too.
    out[n - horizons:] = False
    return out


# ------------------------------------------------------------------------ main

def main() -> None:
    df = pd.read_csv(DATA, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    stamps = pd.DatetimeIndex(df[TIME_COL])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")

    gaps = stamps.to_series().diff().dropna().value_counts()
    print(f"record        {stamps[0]}  ->  {stamps[-1]}   ({len(df):,} rows)")
    print(f"spacing       {dict(list(gaps.items())[:3])}")
    print(f"gate model    publish at {PUBLISH_LOCAL_HOUR:02d}:00 {MARKET_TZ} on D-1, "
          f"delivery day = local calendar day\n")

    leak = leak_matrix(stamps)
    scored = np.zeros(len(df), dtype=bool)
    scored[: len(df) - HORIZONS] = True

    te = (stamps >= pd.Timestamp(TEST_START, tz="UTC")) & (stamps < pd.Timestamp(TEST_END, tz="UTC"))

    for name, mask in (("whole record", scored), ("2025 test window", te & scored)):
        sub = leak[mask]
        cells = sub.size
        print(f"--- {name} ({mask.sum():,} origins, {cells:,} origin-horizon cells) ---")
        print(f"  leaked cells            {sub.sum():>10,}  ({sub.mean():6.2%})")
        rows_hit = sub.any(axis=1)
        print(f"  origins with >=1 leak   {rows_hit.sum():>10,}  ({rows_hit.mean():6.2%})")
        per_row = sub.sum(axis=1)
        print(f"  leaked horizons/origin  mean {per_row.mean():5.1f}   "
              f"max {per_row.max():3d}   (of {HORIZONS})\n")

    # ---- where the leak sits: by horizon and by origin hour ----
    sub = leak[te & scored]
    st = stamps[te & scored]
    by_h = sub.mean(axis=0)
    print("leak rate by horizon (2025)")
    for h in (1, 4, 12, 24, 48, 72, 90, 96):
        print(f"  h={h:<3d} ({h*15/60:5.2f} h ahead)   {by_h[h-1]:6.2%}")
    print(f"  first horizon with any leak: h={int(np.argmax(by_h > 0)) + 1}   "
          f"peak {by_h.max():.2%} at h={int(np.argmax(by_h)) + 1}\n")

    local_hour = st.tz_convert(MARKET_TZ).hour
    print("leak rate by origin hour, local (2025)")
    rows = []
    for hh in range(24):
        m = local_hour == hh
        if m.any():
            rows.append((hh, sub[m].mean(), sub[m].sum(axis=1).mean()))
    for hh, frac, per in rows:
        bar = "#" * int(round(frac * 50))
        print(f"  {hh:02d}:00  {frac:6.2%}  {per:5.1f} horizons  {bar}")
    print()

    # ---- does the leak carry information? ----
    # The honest question is not "is the leak large" but "would the model have known
    # something close to it anyway". The legitimate fallback at an early-morning origin
    # is the price already published for TODAY at the same clock time.
    da = df[DA_COL].to_numpy(dtype="float64")
    n = len(df)
    idx = np.arange(n)
    leaked_vals, fallback_vals = [], []
    te_idx = np.flatnonzero(te & scored)
    for h in range(1, HORIZONS + 1):
        col = leak[:, h - 1]
        hit = np.intersect1d(te_idx, np.flatnonzero(col))
        if hit.size == 0:
            continue
        tgt = hit + h
        # same clock time, previous delivery day = 96 steps earlier
        prev = tgt - HORIZONS
        ok = (prev >= 0) & (da[tgt] != 0.0) & (da[prev] != 0.0)
        leaked_vals.append(da[tgt][ok])
        fallback_vals.append(da[prev][ok])
    lv = np.concatenate(leaked_vals)
    fv = np.concatenate(fallback_vals)
    resid = lv - fv
    print("information content of the leak (2025, both prices published)")
    print(f"  leaked cells compared          {lv.size:,}")
    print(f"  leaked DA price      mean {lv.mean():8.2f}   sd {lv.std():8.2f}")
    print(f"  legitimate fallback  mean {fv.mean():8.2f}   sd {fv.std():8.2f}   "
          f"(same clock time, previous day)")
    print(f"  corr(leaked, fallback)         {np.corrcoef(lv, fv)[0,1]:.4f}")
    print(f"  MAE(fallback vs leaked)        {np.abs(resid).mean():8.2f} EUR/MWh")
    print(f"  R^2 of fallback for leaked     "
          f"{1 - resid.var() / lv.var():.4f}   <- unexplained share is the true leak\n")

    # ---- does the leak help predict the TARGET? this is what decides severity ----
    # A leak only costs you accuracy if the unpublished price predicts the imbalance
    # price better than a legitimate substitute does. Univariate OLS, in-sample: an
    # upper bound on what a model could have extracted from each signal alone.
    price = df[PRICE_COL].to_numpy(dtype="float64")
    y_parts = []
    for h in range(1, HORIZONS + 1):
        hit = np.intersect1d(te_idx, np.flatnonzero(leak[:, h - 1]))
        if hit.size == 0:
            continue
        tgt = hit + h
        prev = tgt - HORIZONS
        ok = (prev >= 0) & (da[tgt] != 0.0) & (da[prev] != 0.0)
        y_parts.append(price[tgt][ok])
    y = np.concatenate(y_parts)

    def ols_r2(x, target):
        x = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        r = target - x @ beta
        return 1 - r.var() / target.var()

    r2_leak, r2_fb = ols_r2(lv, y), ols_r2(fv, y)
    print("predictive value of the leak for the imbalance price at the target")
    print(f"  target imbalance price   mean {y.mean():8.2f}   sd {y.std():8.2f}")
    print(f"  R^2 | leaked DA price          {r2_leak:.4f}")
    print(f"  R^2 | legitimate fallback      {r2_fb:.4f}")
    print(f"  uplift from the leak           {r2_leak - r2_fb:+.4f}\n")

    # ---- what a correct flag would blank ----
    da_lead = np.column_stack([pd.Series(da).shift(-h).to_numpy() for h in range(1, HORIZONS + 1)])
    da_lead = np.nan_to_num(da_lead, nan=0.0)
    present_now = (da_lead != 0.0)
    present_fixed = present_now & ~leak
    for name, mask in (("2025 test window", te & scored),):
        pn, pf = present_now[mask], present_fixed[mask]
        print(f"day-ahead tensor, {name}")
        print(f"  cells flagged present, current code   {pn.sum():>10,}  ({pn.mean():6.2%})")
        print(f"  cells flagged present, gate-correct   {pf.sum():>10,}  ({pf.mean():6.2%})")
        print(f"  cells that must be blanked            {(pn & ~pf).sum():>10,}  "
              f"({(pn & ~pf).mean():6.2%} of all cells, "
              f"{(pn & ~pf).sum() / max(pn.sum(), 1):6.2%} of currently-present cells)")

    # ---- the outage interacts with the gate ----
    dead = (stamps >= pd.Timestamp("2025-10-28", tz="UTC")) & scored
    alive = te & scored & ~dead
    print(f"\n  of the blanked cells, {(present_now & ~present_fixed)[alive].sum():,} fall in "
          f"the DA-alive period and {(present_now & ~present_fixed)[dead & te].sum():,} in the "
          f"post-2025-10-28 outage\n  (the outage rows are already zero, so the gate fix bites "
          f"only where the feed was live)")


if __name__ == "__main__":
    main()
