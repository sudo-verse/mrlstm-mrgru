"""sMAPE and MASE -- answering the supervisor's metric request with numbers.

He asked for sMAPE alongside MAE and RMSE. This module computes it, and computes it
three ways, because the honest answer to the request is not "no" but "here it is, here
is what it does to the ranking, and here is the metric that does the job it was meant
to do".

    sMAPE(y, f) = mean( |f - y| / ((|y| + |f|) / 2) ) * 100

The 2025 test window breaks its assumptions: 6.77% of prices are negative, 640 rows are
exactly zero (the denominator vanishes), and 7.76% lie within +-10 EUR/MWh, where the
denominator is small enough that an ordinary error becomes a huge percentage. sMAPE was
designed to compare forecasts ACROSS series of different scales; there is one series
here, so it is buying nothing and paying a lot.

MASE is the standard replacement: scale the absolute error by the in-sample mean
absolute error of a seasonal-naive forecaster. Zeros and negatives are harmless because
the scaling factor is a property of the training series, not of the individual actual.
MASE < 1 means "better than seasonal naive on average".
"""

import numpy as np


def smape(y_true, y_pred, eps=0.0):
    """Symmetric MAPE in percent. `eps` guards the zero denominator (0 = leave NaN)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    if eps:
        denom = np.maximum(denom, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(y_pred - y_true) / denom
    return float(np.nanmean(np.where(np.isfinite(r), r, np.nan)) * 100)


def smape_restricted(y_true, y_pred, floor=10.0):
    """sMAPE over the rows where |actual| > floor -- the only regime it is meaningful in.

    Reporting this next to the unrestricted figure IS the argument: the gap between them
    is entirely produced by intervals where the denominator collapsed, not by any
    difference in forecast quality.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = np.abs(y_true) > floor
    return smape(y_true[m], y_pred[m]), int(m.sum()), float(m.mean())


def mase_scale(train_series, season=96):
    """In-sample seasonal-naive MAE -- the denominator of MASE. Season = one day."""
    s = np.asarray(train_series, dtype="float64")
    s = s[~np.isnan(s)]
    d = np.abs(s[season:] - s[:-season])
    return float(np.mean(d))


def mase(y_true, y_pred, scale):
    """Mean absolute scaled error. < 1 beats seasonal naive; handles zeros and negatives."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_pred - y_true)) / scale)


def sensitivity_table(errors=(5.0,), actuals=(200.0, 20.0, 2.0, -3.0)):
    """A fixed absolute error, scored by sMAPE at several price levels.

    This is the demonstration to put in front of a supervisor: the SAME forecast quality
    scores anywhere from a couple of percent to 200% depending only on where the price
    happens to sit.
    """
    rows = []
    for e in errors:
        for a in actuals:
            f = a + e
            rows.append({"actual": a, "forecast": f, "abs_error": abs(e),
                         "sMAPE_%": smape([a], [f])})
    return rows


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("/home/sudoverse/fyp/imbalance_data.csv",
                     parse_dates=["Time [UTC] start"])
    te = df[(df["Time [UTC] start"] >= "2025-01-01") &
            (df["Time [UTC] start"] < "2026-01-01")]
    p = te["imbalance_price"].to_numpy()

    print("2025 test window -- why sMAPE misfires here")
    print(f"  rows                 {len(p):,}")
    print(f"  negative prices      {(p < 0).mean():6.2%}   ({(p < 0).sum():,})")
    print(f"  exactly zero         {(p == 0).sum():,}   <- sMAPE undefined")
    print(f"  |price| <= 10        {(np.abs(p) <= 10).mean():6.2%}   "
          f"({(np.abs(p) <= 10).sum():,})   <- denominator collapses\n")

    print("a fixed 5 EUR/MWh error, scored by sMAPE")
    for r in sensitivity_table():
        print(f"  actual {r['actual']:>7.0f}   forecast {r['forecast']:>7.0f}   "
              f"sMAPE {r['sMAPE_%']:7.1f}%")
    print("\n  same forecast quality, 2.5% to 200% depending only on the price level.")
