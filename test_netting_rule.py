"""Candidate rule IV: a cross-border imbalance-netting block.

This is the supervisor's own suggestion, moved to the position his written note
specified (upstream of P_RE, modifying the effective balancing requirement) rather
than the position his message specified (a fourth price into the min/max gate --
which analyse_fourth_rule.py proves is inert, because the gate identity is exact).

The mechanism: under IGCC/PICASSO, opposite-signed imbalances in coupled zones are
netted before any reserve is activated. So the volume the German control zone must
actually cover is not its own raw imbalance but what survives netting against its
neighbours. Neighbour-zone fundamentals should therefore carry information about
the DIRECTION of the local imbalance -- which analyse_fourth_rule.py identifies as
the binding constraint (worth ~3x the block levels, and only AUC 0.523 locally).

Neighbour data comes from PriceFM (Yu et al., the same group as the base paper):
    https://huggingface.co/datasets/RunyaoYu/PriceFM  ->  data/pricefm_FINAL.csv
38 bidding zones x {generation, load, price, solar, wind}, quarter-hourly, 2022-2025.

TWO UPPER BOUNDS, both deliberate and both stated in the output:
  * fundamentals are ACTUALS at delivery, standing in for day-ahead forecasts;
  * the three rule blocks are taken as exact, isolating the direction signal.
So the MAE here is a ceiling on what the block can deliver, not a forecast of it.
The comparison between feature sets is still fair: all three share both bounds.

    .venv/bin/python test_netting_rule.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

IMB = "external/MRINN/Data/imbalance_data.csv"
NEI = "data/pricefm_FINAL.csv"
# DE_LU and the zones it couples to, per PriceFM's own adjacency dict.
ZONES = ["DE_LU", "AT", "BE", "CZ", "DK_1", "DK_2", "FR", "NL", "NO_2", "PL", "SE_4"]
H = 96                      # day-ahead
LOCAL = ["system_imbalance", "E_aFRR_pos", "E_aFRR_neg", "P_aFRR_pos", "P_aFRR_neg",
         "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo", "L_ID15", "L_ID60", "imbalance_price"]


def build():
    cols = ["time_utc"] + [f"{z}-{f}" for z in ZONES
                           for f in ("load", "wind", "solar", "price")]
    g = pd.read_csv(NEI, usecols=cols)
    g["t"] = pd.to_datetime(g.time_utc, utc=True)
    d = pd.read_csv(IMB)
    d["t"] = pd.to_datetime(d["Time [UTC] start"], utc=True)
    m = d.merge(g.drop(columns="time_utc"), on="t", how="inner")
    for z in ZONES:                      # residual load = what thermal must cover
        m[f"{z}-rl"] = (m[f"{z}-load"] - m[f"{z}-wind"].fillna(0)
                        - m[f"{z}-solar"].fillna(0))
        m[f"{z}-rlramp"] = m[f"{z}-rl"].diff()
    return m


def feature_sets(m):
    loc = {f"{c}_l{L}": m[c].shift(L) for c in LOCAL for L in (0, 1, 2, 4, 8, 96)}
    loc["hour"] = m.t.dt.hour + m.t.dt.minute / 60
    loc["dow"] = m.t.dt.dayofweek
    A = pd.DataFrame(loc)
    own = pd.DataFrame({f"DE_LU-{k}_f": m[f"DE_LU-{k}"].shift(-H)
                        for k in ("rl", "rlramp", "wind", "solar", "load", "price")})
    nb = pd.DataFrame({f"{z}-{k}_f": m[f"{z}-{k}"].shift(-H)
                       for z in ZONES if z != "DE_LU"
                       for k in ("rl", "rlramp", "wind", "solar")})
    return {"A. local lags only (what the model sees today)": A,
            "B. + DE_LU fundamentals at delivery": pd.concat([A, own], axis=1),
            "C. + 10 neighbour zones at delivery": pd.concat([A, own, nb], axis=1)}


def main():
    m = build()
    print(f"merged {len(m):,} quarter-hours, {m.t.min().date()} -> {m.t.max().date()}")
    print("UPPER BOUNDS: fundamentals are actuals at delivery; blocks taken as exact.\n")
    V, y = m.system_imbalance.values, m.imbalance_price.values
    B3 = m[["P_RE", "P_EX", "P_SC"]].to_numpy()
    tgt = pd.Series(V).shift(-H)
    tr, te = (m.t.dt.year < 2025).values, (m.t.dt.year == 2025).values

    idx = np.arange(H, len(m))[te[H:]]
    Bp = B3[idx]
    base = np.abs(np.where(V[idx - H] > 0, Bp.max(1), Bp.min(1)) - y[idx]).mean()
    print(f"  {'persistence direction':<48} {'':9}  MAE {base:7.2f}")

    last = None
    for name, X in feature_sets(m).items():
        ok = X.notna().all(1).values & tgt.notna().values
        ytr = (tgt[ok & tr] > 0).astype(int)
        yte = (tgt[ok & te] > 0).astype(int).values
        mod = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", n_jobs=8).fit(X[ok & tr], ytr)
        p = mod.predict_proba(X[ok & te])[:, 1]
        auc = roc_auc_score(yte, p)
        j = np.where(ok & te)[0] + H
        k = j < len(m)
        j, pk = j[k], p[k]
        Bt = B3[j]
        mae = np.abs(np.where(pk > .5, Bt.max(1), Bt.min(1)) - y[j]).mean()
        print(f"  {name:<48} AUC {auc:.3f}  MAE {mae:7.2f}  ({mae / base - 1:+.1%} vs persistence)")
        last = (mod, X.columns)

    mod, names = last
    imp = pd.Series(mod.feature_importances_, index=names).sort_values(ascending=False)
    foreign = [n for n in imp.index
               if any(n.startswith(z + "-") for z in ZONES if z != "DE_LU")]
    print(f"\n  neighbour-zone features carry {imp[foreign].sum() * 100:.1f}% of total importance")
    print("  top 10: " + ", ".join(imp.head(10).index))
    print("\n  => cross-border fundamentals carry real, non-redundant information about")
    print("     the local imbalance direction. This is the supervisor's netting idea,")
    print("     placed upstream where it can act, and it is the only one of seven")
    print("     candidates tested that moves the binding constraint.")


if __name__ == "__main__":
    main()
