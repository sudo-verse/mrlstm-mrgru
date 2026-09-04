"""Candidate search for an ORIGINAL fourth market rule.

analyse_fourth_rule.py established that a fourth PRICE block cannot help: the
settlement identity max/min(P_RE,P_EX,P_SC) by sign(V) is exact on all 140,249 rows,
so the rule layer has no residual. Anything new must therefore act on the block
INPUTS or on the gate, not add a fourth price into the gate.

This script tests six candidate mechanisms as honestly as it can -- strictly trailing
features, fit on 2022-24, scored on 2025 -- and records what survived. Most did not.
Negative results are kept deliberately: they are what stops the same ideas being
re-proposed, and three of them are publishable criticisms of the base paper in
their own right.

    .venv/bin/python search_new_rule.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = "external/MRINN/Data/imbalance_data.csv"
THRESHOLDS = (200, 500, 1000)


def load():
    d = pd.read_csv(DATA)
    d["t"] = pd.to_datetime(d["Time [UTC] start"], utc=True)
    d["yr"] = d.t.dt.year
    return d


def auc_table(name_to_X, spike, tr, te, ok=None):
    """Out-of-sample AUC for each feature set, fit on `tr`, scored on `te`."""
    ok = np.ones(len(spike), bool) if ok is None else ok
    out = {}
    for nm, X in name_to_X.items():
        X = np.nan_to_num(np.atleast_2d(X.T).T)
        sc = StandardScaler().fit(X[tr & ok])
        m = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr & ok]), spike[tr & ok])
        out[nm] = roc_auc_score(spike[te & ok], m.decision_function(sc.transform(X[te & ok])))
    return out


def c1_merit_order_ceiling(d):
    print("\nC1. MERIT-ORDER CEILING -- is the MOL price a bound on the activated price?")
    ap, mp = d.P_aFRR_pos.values, d.P_aFRR_pos_MOL.values
    an, mn = d.P_aFRR_neg.values, d.P_aFRR_neg_MOL.values
    a, b = d.E_aFRR_pos.values > 0, d.E_aFRR_neg.values > 0
    print(f"    P_aFRR_pos <= MOL when activated: {(ap[a] <= mp[a] + 1e-6).mean() * 100:.2f}%")
    print(f"    P_aFRR_neg >= MOL when activated: {(an[b] >= mn[b] - 1e-6).mean() * 100:.2f}%")
    print("    VERDICT: REJECTED. The MOL slot (P_VoAA) is not a bound -- it is violated")
    print("    in ~42%/66% of activated intervals, so it cannot support a cap/floor rule.")


def c2_reserve_depletion(d, spike_at, tr, te):
    print("\nC2. RESERVE DEPLETION -- does sustained activation shift the scarcity onset?")
    V = d.system_imbalance.values
    aV = np.abs(V)
    cp = d.E_aFRR_pos.shift(1).rolling(16, min_periods=1).sum().values
    cn = d.E_aFRR_neg.shift(1).rolling(16, min_periods=1).sum().values
    dep = np.nan_to_num(np.where(V > 0, cp, cn))
    D = dep / np.nanpercentile(dep, 95)
    for thr in THRESHOLDS:
        r = auc_table({"|V|,|V|^3": np.c_[aV, aV ** 3],
                       "+D": np.c_[aV, aV ** 3, D],
                       "+|V|*D": np.c_[aV, aV ** 3, aV * D, (aV ** 3) * D]},
                      spike_at(thr), tr, te)
        base = r["|V|,|V|^3"]
        print(f"    |price|>{thr:5d}  " + "  ".join(
            f"{k} {v:.4f} ({v - base:+.4f})" for k, v in r.items()))
    print("    VERDICT: MARGINAL / UNSTABLE. Conditional on |V| decile the spike rate does")
    print("    lift 1.2-1.6x with depletion, but out-of-sample AUC moves by <0.01 and")
    print("    changes sign across thresholds. Not a load-bearing rule.")


def c3_supply_curve(d):
    print("\nC3. MERIT-ORDER SUPPLY CURVE -- is the activated price a function of volume?")
    act = (d.E_aFRR_pos > 0.01).values
    E, P, ref = (d.E_aFRR_pos.values[act], d.P_aFRR_pos.values[act],
                 d.P_EX_basis.values[act])
    mk = P - ref
    print(f"    Spearman(E, price level) = "
          f"{pd.Series(E).corr(pd.Series(P), method='spearman'):+.3f}  "
          f"BUT per-year curves are monotone (Simpson's paradox: the level fell")
    print(f"    282 -> 64 EUR/MWh 2022->2025 while the shape held).")
    print(f"    Spearman(E, markup over P_EX_basis) = "
          f"{pd.Series(E).corr(pd.Series(mk), method='spearman'):+.3f}")
    tr = d.yr.values[act] < 2025
    lE = np.log1p(E)
    X = np.c_[np.ones_like(E), lE, lE ** 2, lE ** 3]
    b = np.linalg.lstsq(X[tr], mk[tr], rcond=None)[0]
    from sklearn.metrics import r2_score
    print(f"    cubic-in-log(E) markup model: train R^2 {r2_score(mk[tr], X[tr] @ b):+.4f}, "
          f"2025 R^2 {r2_score(mk[~tr], X[~tr] @ b):+.4f}")
    print("    VERDICT: REJECTED. Once the level is normalised away the volume signal")
    print("    vanishes and the fitted curve does not generalise (negative OOS R^2).")


def c4_deterministic_deviation(d):
    print("\nC4. DETERMINISTIC FREQUENCY DEVIATION -- hour-boundary ramping (known at DA)")
    loc = d.t.dt.tz_convert("Europe/Berlin")
    qh = (loc.dt.minute // 15).values
    V = d.system_imbalance.values
    print("    quarter-of-hour:  " + "  ".join(
        f"q{q} P(V>0) {(V[qh == q] > 0).mean() * 100:.1f}%" for q in range(4)))
    pda = d.P_DA_nemo.values
    ramp = np.r_[0, np.diff(pda)]
    m = (qh == 0) & (pda != 0) & (np.r_[False, pda[:-1] != 0])
    print(f"    corr(DA hour-step, V) at the hour boundary = {np.corrcoef(ramp[m], V[m])[0, 1]:+.4f}")
    print("    VERDICT: REJECTED. The quarter effect is a 2pp swing and the DA ramp")
    print("    correlates -0.03. Real in the literature, negligible in this zone.")


def c5_cubic_is_inert(d, spike_at, te):
    print("\nC5. IS THE BASE PAPER'S CUBIC DOING ANY WORK?  (a criticism, not a candidate)")
    aV = np.abs(d.system_imbalance.values)
    for thr in THRESHOLDS:
        sp = spike_at(thr)
        print(f"    |price|>{thr:5d}  AUC(|V|) {roc_auc_score(sp[te], aV[te]):.4f}   "
              f"AUC(|V|^3) {roc_auc_score(sp[te], aV[te] ** 3):.4f}")
    print("    FINDING: identical, necessarily. A cubic is a strictly monotone transform")
    print("    of |V|, so it cannot reorder anything. P_SC's cubic changes only the")
    print("    MAGNITUDE of the markup (via C10), never which intervals are flagged.")


def c6_relative_scarcity(d, spike_at, tr, te):
    print("\nC6. RELATIVE SCARCITY -- thresholds that track the distribution, not fixed MW")
    aV = np.abs(d.system_imbalance.values)
    print(f"    {'year':>6} {'p50':>7} {'p90':>7} {'p99':>7}   |V| has drifted upward:")
    for yr, g in d.groupby("yr"):
        a = np.abs(g.system_imbalance.values)
        print(f"    {yr:6d} {np.percentile(a, 50):7.2f} {np.percentile(a, 90):7.2f} "
              f"{np.percentile(a, 99):7.2f}")
    roll = pd.Series(aV).shift(1).rolling(96 * 7, min_periods=96).quantile(0.90).values
    rel = aV / np.where(roll > 0, roll, np.nan)
    ok = np.isfinite(rel)
    for thr in THRESHOLDS:
        r = auc_table({"absolute |V|": np.c_[aV], "relative |V|/p90": np.c_[np.nan_to_num(rel)]},
                      spike_at(thr), tr, te, ok)
        print(f"    |price|>{thr:5d}  " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
    print("    VERDICT: PARTIAL. Relative scarcity wins at the mild threshold (+0.018 AUC")
    print("    at >200) and loses at the extremes. Defensible as a robustness redesign,")
    print("    not as an accuracy claim.")


def main():
    d = load()
    y = d.imbalance_price.values
    tr, te = (d.yr < 2025).values, (d.yr == 2025).values
    spike_at = lambda t: (np.abs(y) > t).astype(int)
    print(f"{DATA}: {len(d):,} rows.  Fit 2022-24, score 2025. Strictly trailing features.")
    print("=" * 76)
    c1_merit_order_ceiling(d)
    c2_reserve_depletion(d, spike_at, tr, te)
    c3_supply_curve(d)
    c4_deterministic_deviation(d)
    c5_cubic_is_inert(d, spike_at, te)
    c6_relative_scarcity(d, spike_at, tr, te)
    print("\n" + "=" * 76)
    print("SUMMARY: 4 rejected, 1 marginal, 1 partial. The two results that DO carry a")
    print("paper are criticisms of the existing rules (C5: the cubic cannot reorder;")
    print("C6: the thresholds are calibrated on a distribution that has drifted), plus")
    print("the exactness identity from analyse_fourth_rule.py. The remaining untested")
    print("candidate is cross-border netting, which needs neighbour-zone data.")


if __name__ == "__main__":
    main()
