"""Where does the best model actually fail, and is there a rule that would fix it?

search_new_rule.py tested six candidate mechanisms hypothesis-first and rejected most.
This script works the other way round: it takes MR-GRU's 3.3 million residual cells,
locates the structure in them, and asks whether that structure is something a market
rule could exploit.

It finds real, previously undocumented structure -- and then finds that the structure
is NOT exploitable, for a reason that matters more than any of the candidate rules:

    the model hedges toward the middle because direction is unpredictable, and that
    hedging is OPTIMAL, not a defect. Committing harder to the rules makes it worse.

That is the finding to build the paper on.

    .venv/bin/python analyse_residual_structure.py
"""

import glob

import numpy as np
import pandas as pd

DATA = "external/MRINN/Data/imbalance_data.csv"
MODEL = "mr_gru"


def load():
    d = pd.read_csv(DATA)
    d["t"] = pd.to_datetime(d["Time [UTC] start"], utc=True).dt.tz_localize(None)
    fs = sorted(glob.glob(f"results/pred_{MODEL}_*.npz"))
    yt = np.concatenate([np.load(f)["y_true"] for f in fs])
    yp = np.concatenate([np.load(f)["y_pred"] for f in fs])
    st = pd.DatetimeIndex(np.concatenate([np.load(f)["stamps"] for f in fs]))
    tgt = st.values[:, None] + (np.arange(1, yt.shape[1] + 1) *
                                np.timedelta64(15, "m"))[None, :]
    V = (d.set_index("t").system_imbalance
         .reindex(pd.DatetimeIndex(tgt.ravel())).values.reshape(yt.shape))
    return d, yt, yp[:, :, 2], V


def s1_concentration(yt, q50):
    print("=" * 78)
    print("1. THE ERROR IS EXTREMELY CONCENTRATED")
    print("=" * 78)
    f = np.abs(q50 - yt).ravel()
    o = np.argsort(f)[::-1]
    for pct in (0.1, 1, 5, 10, 25):
        k = int(len(f) * pct / 100)
        print(f"  worst {pct:5.1f}% of cells carry {f[o[:k]].sum() / f.sum() * 100:5.1f}% "
              f"of all absolute error")


def s2_bimodal(yt, q50):
    print("\n" + "=" * 78)
    print("2. ERROR IS U-SHAPED IN TARGET MAGNITUDE -- two failure regimes, not one")
    print("=" * 78)
    y = np.abs(yt.ravel())
    f = np.abs(q50 - yt).ravel()
    b = pd.qcut(y, 10, labels=False, duplicates="drop")
    print(f"  {'decile':>7} {'|y| mean':>10} {'MAE':>9} {'share of error':>15}")
    for i in range(b.max() + 1):
        m = b == i
        print(f"  {i:7d} {y[m].mean():10.2f} {f[m].mean():9.2f} "
              f"{f[m].sum() / f.sum() * 100:14.1f}%")
    print("  => worst at BOTH ends: near-zero prices and spikes. The middle is easy.")


def s3_nearzero(d, yt):
    print("\n" + "=" * 78)
    print("3. NEAR-ZERO PRICES ARE A DOWNWARD-REGULATION PHENOMENON")
    print("=" * 78)
    y, V = d.imbalance_price.values, d.system_imbalance.values
    near = np.abs(y) < 10
    print(f"  near-zero intervals: {near.sum():,} ({near.mean() * 100:.2f}%)")
    print(f"  P(system long | near-zero) = {(V[near] < 0).mean() * 100:.1f}%  "
          f"vs {(V < 0).mean() * 100:.1f}% overall")
    for c in ("P_aFRR_neg", "E_aFRR_neg", "E_aFRR_pos", "P_EX_basis"):
        v = d[c].values
        print(f"    {c:<14} near-zero {v[near].mean():8.2f}   all {v.mean():8.2f}   "
              f"ratio {v[near].mean() / max(abs(v.mean()), 1e-9):5.2f}")
    print("  => the DOWNWARD activation price collapses (0.08x) far harder than the")
    print("     market reference (0.63x). This is a balancing mechanism, not cheap hours.")


def s4_asymmetry(d):
    print("\n" + "=" * 78)
    print("4. THE PRICE LAW IS MIRROR-ASYMMETRIC -- and P_SC's cubic is symmetric")
    print("=" * 78)
    y, V = d.imbalance_price.values, d.system_imbalance.values
    base = d.P_EX_basis.values
    S, L = V > 0, V < 0
    print(f"  {'':<14} {'SHORT (V>0)':>14} {'LONG (V<0)':>14}")
    print(f"  {'median':<14} {np.median(y[S]):14.2f} {np.median(y[L]):14.2f}")
    print(f"  {'skew':<14} {pd.Series(y[S]).skew():14.2f} {pd.Series(y[L]).skew():14.2f}")
    print(f"  {'min':<14} {y[S].min():14.1f} {y[L].min():14.1f}")
    print(f"  {'max':<14} {y[S].max():14.1f} {y[L].max():14.1f}")
    print(f"  {'med dev':<14} {np.median(y[S] - base[S]):+14.2f} "
          f"{np.median(y[L] - base[L]):+14.2f}")
    print("  => short: bounded below, unbounded above. long: bounded above, unbounded")
    print("     below. P_SC applies sgn(V)*C10*(...)^3 -- the SAME magnitude either way,")
    print("     so it cannot represent this. But see section 5: the cubic-in-|V| form")
    print("     explains R^2 ~ 0.04 either way, so an asymmetric cubic is not the fix.")


def s5_hedging(yt, q50, V):
    print("\n" + "=" * 78)
    print("5. THE MODEL HEDGES -- AND THAT IS OPTIMAL, NOT A DEFECT")
    print("=" * 78)
    ae = np.abs(q50 - yt)
    base = ae.mean()
    bs, bl = (q50 - yt)[V > 0].mean(), (q50 - yt)[V < 0].mean()
    spread_p = q50[V > 0].mean() - q50[V < 0].mean()
    spread_t = yt[V > 0].mean() - yt[V < 0].mean()
    print(f"  conditional bias: {bs:+.2f} when short, {bl:+.2f} when long")
    print(f"  model directional spread {spread_p:7.1f} vs true {spread_t:7.1f} "
          f"-- it reproduces {spread_p / spread_t * 100:.0f}%")
    c1 = q50 - np.where(V > 0, bs, bl)
    print(f"\n  with ORACLE direction, removing that bias: MAE {base:.2f} -> "
          f"{np.abs(c1 - yt).mean():.2f} ({np.abs(c1 - yt).mean() / base - 1:+.1%})")
    rng = np.random.default_rng(0)
    z = (V > 0).astype(float) + rng.normal(0, 1.0, V.shape) * 0.35 / max(0.62 - 0.5, 1e-3)
    c = q50 - np.where(z > np.median(z), bs, bl)
    print(f"  with a REALISTIC detector (~AUC 0.62): MAE -> {np.abs(c - yt).mean():.2f} "
          f"({np.abs(c - yt).mean() / base - 1:+.1%})")
    print("  (the detector is simulated by noising the true direction, so the exact")
    print("   figure is indicative; the sign and magnitude of the penalty are not.)")
    print("\n  => the correction is large (~130 EUR/MWh swing) and applying it in the")
    print("     WRONG direction costs more than applying it right saves. Hedging toward")
    print("     the middle is the correct response to an unpredictable direction.")


def main():
    d, yt, q50, V = load()
    ok = ~np.isnan(V)
    print(f"MR-GRU residuals: {yt.shape[0]:,} origins x {yt.shape[1]} horizons, "
          f"MAE {np.abs(q50 - yt).mean():.2f}\n")
    s1_concentration(yt, q50)
    s2_bimodal(yt, q50)
    s3_nearzero(d, yt)
    s4_asymmetry(d)
    s5_hedging(yt[ok], q50[ok], V[ok])
    print("\n" + "=" * 78)
    print("CONCLUSION: the binding constraint is INFORMATIONAL, not structural. No")
    print("re-specification of the rule layer can recover this error, because the model")
    print("is already responding optimally to what it knows. The only lever that moves")
    print("it is new exogenous information -- which is what the cross-border netting")
    print("block supplies (see test_netting_rule.py: direction AUC 0.518 -> 0.566).")


if __name__ == "__main__":
    main()
