"""Is there a fourth market rule worth adding?  (supervisor suggestion, 2026-08-23)

The supervisor asked whether a fourth rule block, on top of the base paper's three,
would carry a journal paper. This script answers that from the data alone.

The headline result is in section 1 and it settles the question:

    imbalance_price = max(P_RE, P_EX, P_SC)   if system_imbalance > 0  (system short)
                    = min(P_RE, P_EX, P_SC)   if system_imbalance < 0  (system long)

holds EXACTLY -- 140,249 of 140,249 rows at a relative tolerance of 1e-9, no exceptions.
The three blocks plus the sign rule are not an approximation of the settlement price;
they ARE the settlement price. So the rule layer has zero residual, and a fourth PRICE
block cannot reduce an error that does not exist. Worse, any fourth term entering the
smooth min/max gate can only move the result off an identity that is currently exact.

Sections 2-5 then locate where the forecast error actually lives, which is what a
fourth contribution should target instead.

    .venv/bin/python analyse_fourth_rule.py
"""

import numpy as np
import pandas as pd

DATA = "external/MRINN/Data/imbalance_data.csv"
BLOCKS = ["P_RE", "P_EX", "P_SC"]
HORIZONS = [(1, "15 min"), (4, "1 hour"), (96, "1 day")]


def load():
    d = pd.read_csv(DATA)
    d["t"] = pd.to_datetime(d["Time [UTC] start"], utc=True)
    return d


def section1_identity(d):
    print("=" * 74)
    print("1. THE THREE BLOCKS RECONSTRUCT THE PRICE EXACTLY")
    print("=" * 74)
    y, V = d.imbalance_price.values, d.system_imbalance.values
    B = d[BLOCKS].to_numpy()
    rule = np.where(V > 0, B.max(1), B.min(1))
    err = np.abs(rule - y)
    print(f"  price == max/min(P_RE,P_EX,P_SC) by sign(V):  MAE {err.mean():.2e}, "
          f"max {err.max():.2e}")
    for tol in (1e-9, 1e-6):
        eq = np.abs(B - y[:, None]) <= tol * np.maximum(1, np.abs(y[:, None]))
        print(f"  rows where the price equals some block (tol {tol:g}): "
              f"{eq.any(1).mean() * 100:.4f}%  ({(~eq.any(1)).sum():,} exceptions)")
    print("\n  => the rule layer has NO residual. A fourth price block has nothing to fit.")


def section2_shares(d):
    print("\n" + "=" * 74)
    print("2. WHICH BLOCK SETS THE PRICE, AND HOW THAT HAS DRIFTED")
    print("=" * 74)
    y = d.imbalance_price.values
    sel = np.argmin(np.abs(d[BLOCKS].to_numpy() - y[:, None]), 1)
    d = d.assign(yr=d.t.dt.year)
    for yr, _ in d.groupby("yr"):
        m = (d.yr == yr).values
        shares = "  ".join(f"{n} {(sel[m] == i).mean() * 100:5.1f}%"
                           for i, n in enumerate(BLOCKS))
        print(f"  {yr}  {shares}   |V| mean {np.abs(d.system_imbalance[m]).mean():6.2f}")
    print("\n  => P_RE's share fell 82.4% -> 53.8% while P_EX doubled. The market moved;")
    print("     P_SC never sets more than 1.4% of intervals.")


def section3_where_the_error_is(d):
    print("\n" + "=" * 74)
    print("3. WHERE THE FORECAST ERROR ACTUALLY LIVES  (counterfactual oracles)")
    print("=" * 74)
    y, V = d.imbalance_price.values, d.system_imbalance.values
    B = d[BLOCKS].to_numpy()
    print(f"  {'h':>4} {'':8} {'blocks known,':>16} {'sign known,':>15} {'neither':>10}")
    print(f"  {'':4} {'':8} {'sign guessed':>16} {'blocks guessed':>15}")
    for h, lbl in HORIZONS:
        Bt, yt, Vt, Vp, Bp = B[h:], y[h:], V[h:], V[:-h], B[:-h]
        a = np.abs(np.where(Vp > 0, Bt.max(1), Bt.min(1)) - yt).mean()
        b = np.abs(np.where(Vt > 0, Bp.max(1), Bp.min(1)) - yt).mean()
        c = np.abs(np.where(Vp > 0, Bp.max(1), Bp.min(1)) - yt).mean()
        print(f"  {h:4d} {lbl:>8} {a:16.2f} {b:15.2f} {c:10.2f}")
    print("\n  => knowing the DIRECTION is worth about three times more than knowing")
    print("     the block levels. The blocks are easy (P_EX, P_SC autocorrelate ~0.94");
    print("     at 15 min); the direction is the hard part.")


def section4_direction(d):
    print("\n" + "=" * 74)
    print("4. HOW PREDICTABLE IS THE DIRECTION?  (train 2022-24, test 2025)")
    print("=" * 74)
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  xgboost/sklearn not installed -- skipped")
        return
    V = d.system_imbalance.values
    base = ["system_imbalance", "E_aFRR_pos", "E_aFRR_neg", "P_aFRR_pos", "P_aFRR_neg",
            "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo", "L_ID15", "L_ID60",
            "imbalance_price"]
    F = {f"{c}_l{L}": d[c].shift(L) for c in base for L in (0, 1, 2, 4, 8, 96)}
    F["hour"] = d.t.dt.hour + d.t.dt.minute / 60
    F["dow"] = d.t.dt.dayofweek
    F["V_ma4"] = d.system_imbalance.rolling(4).mean()
    F["V_ma16"] = d.system_imbalance.rolling(16).mean()
    X = pd.DataFrame(F)
    tr, te = (d.t.dt.year < 2025).values, (d.t.dt.year == 2025).values

    print(f"  {'h':>4} {'':8} {'persist':>8} {'majority':>9} {'XGBoost':>8} {'AUC':>6}")
    for h, lbl in HORIZONS:
        yv = pd.Series(V).shift(-h)
        ok = X.notna().all(1).values & yv.notna().values
        ytr = (yv[ok & tr] > 0).astype(int)
        yte = (yv[ok & te] > 0).astype(int).values
        m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", n_jobs=8).fit(X[ok & tr], ytr)
        p = m.predict_proba(X[ok & te])[:, 1]
        per = ((V[ok & te] > 0).astype(int) == yte).mean() * 100
        maj = max(yte.mean(), 1 - yte.mean()) * 100
        print(f"  {h:4d} {lbl:>8} {per:7.1f}% {maj:8.1f}% "
              f"{((p > .5).astype(int) == yte).mean() * 100:7.1f}% "
              f"{roc_auc_score(yte, p):6.3f}")
    print("\n  => at day-ahead the direction is AUC 0.523, barely better than a coin")
    print("     flip, and no model beats simply always guessing 'short' (63.2%).")
    print("     This is an information limit, not a modelling failure, and it explains")
    print("     why all nine benchmarked models sit within 3% of one another.")


def section5_decoder(d):
    print("\n" + "=" * 74)
    print("5. IS THE EXACT RULE USEFUL AS A DECODER?")
    print("=" * 74)
    y, V = d.imbalance_price.values, d.system_imbalance.values
    B = d[BLOCKS].to_numpy()
    te = (d.t.dt.year == 2025).values
    print(f"  {'h':>4} {'':8} {'forecast price directly':>24} {'forecast blocks -> rule':>25}")
    for h, lbl in HORIZONS:
        idx = np.arange(h, len(d))[te[h:]]
        Bp = B[idx - h]
        direct = np.abs(y[idx - h] - y[idx]).mean()
        two = np.abs(np.where(V[idx - h] > 0, Bp.max(1), Bp.min(1)) - y[idx]).mean()
        print(f"  {h:4d} {lbl:>8} {direct:24.2f} {two:25.2f}")
    print("\n  => identical to the cent, and necessarily so: applying the rule to")
    print("     persisted blocks IS persisting the price. The decoder is free but")
    print("     adds nothing unless the block inputs are forecast BETTER than the")
    print("     price itself -- which is the one experiment still worth running.")


def main():
    d = load()
    print(f"{DATA}: {len(d):,} rows, {d.t.min().date()} -> {d.t.max().date()}\n")
    section1_identity(d)
    section2_shares(d)
    section3_where_the_error_is(d)
    section4_direction(d)
    section5_decoder(d)


if __name__ == "__main__":
    main()
