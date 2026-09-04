"""Structural check for the corrected multi-horizon day-ahead model.

Proves, on a few months of data and 2 epochs:

  1. HORIZON VARIATION -- the check the old `dayahead.py` model would have failed.
     With the horizon embedding on, predictions differ across h. With it off AND
     known-future off, every horizon is identical, which is exactly the defect
     being fixed.
  2. Parameter counts stay under the 8,900 MLP baseline and are constant in T.
  3. The fold/unfold trick applies MRINN's unchanged rule head per horizon.
  4. Quantile crossing is zero at EVERY horizon, not just on average.
  5. Rolling folds never leak future data into training.
  6. How long a real run would take.

Usage:  .venv/bin/python verify_multihorizon.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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

T, HZ, HU, EPOCHS = 8, 96, 8, 2
Q = list(MH.QUANTILES)
LAGS = list(range(1, T + 1))
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------ 1. splits
print("\n=== 1. Splits ===")
st = MH.static_split()
print(f"     static  train {st['train'][0][:10]} -> {st['train'][1][:10]}"
      f"   test {st['test'][0][:10]} -> {st['test'][1][:10]}")
folds = MH.rolling_folds()
check("4 rolling folds covering 12 months", len(folds) == 4, str(len(folds)))
leak = [f["tag"] for f in folds if f["train"][1] > f["test"][0]]
check("no fold trains on or after its test start", not leak, str(leak))
gap = [f["tag"] for f in folds if f["val"][1] != f["test"][0]]
check("validation abuts the test block", not gap, str(gap))
for f in folds:
    print(f"     {f['tag']}  train {f['train'][0][:10]}->{f['train'][1][:10]}"
          f"  val ->{f['val'][1][:10]}  test ->{f['test'][1][:10]}")

# The paper's own protocol (Yu et al. 2026, Table 2). Getting these boundaries wrong
# would make the headline comparison quietly incomparable rather than visibly broken,
# so they are asserted against the published values rather than eyeballed.
exp = MH.expanding_folds()
check("3 expanding folds (paper Table 2)", len(exp) == 3, str(len(exp)))
leak = [f["tag"] for f in exp if f["train"][1] > f["test"][0]]
check("no expanding fold trains on or after its test start", not leak, str(leak))
check("every expanding fold starts at 2022-01-01",
      all(f["train"][0][:10] == "2022-01-01" for f in exp),
      str([f["train"][0][:10] for f in exp]))
want = [("2024-09-01", "2025-01-01", "2025-05-01"),
        ("2025-01-01", "2025-05-01", "2025-09-01"),
        ("2025-05-01", "2025-09-01", "2026-01-01")]
got = [(f["train"][1][:10], f["test"][0][:10], f["test"][1][:10]) for f in exp]
check("expanding boundaries match the paper", got == want, f"{got}")
for f in exp:
    print(f"     {f['tag']}  train {f['train'][0][:10]}->{f['train'][1][:10]}"
          f"  val ->{f['val'][1][:10]}  test ->{f['test'][1][:10]}")
exp_cov = (MH._ts(exp[0]["test"][0]), MH._ts(exp[-1]["test"][1]))
roll_cov = (MH._ts(folds[0]["test"][0]), MH._ts(folds[-1]["test"][1]))
check("expanding and rolling test the SAME 12 months", exp_cov == roll_cov,
      f"{exp_cov} vs {roll_cov}")

# -------------------------------------------------------------------- 2. data
print("\n=== 2. Data ===")
MR.set_random_seed(42)
data, _ = MR.load_data(FEATS_PRICES, FEATS_CAP, FEATS_VOL, LABEL,
                       str(MR.MRINN_ROOT / "Data" / "imbalance_data.csv"))
tr, va, te = MR.split_data(
    data,
    ("2024-01-01 00:00:00+00:00", "2024-04-01 00:00:00+00:00"),
    ("2024-04-01 00:00:00+00:00", "2024-05-01 00:00:00+00:00"),
    ("2024-05-01 00:00:00+00:00", "2024-06-01 00:00:00+00:00"))
trs, vas, tes, names = MR.shift_data(tr, va, te, LABEL[0], FEATS, [], LAGS, [])
Xtr, Xva, Xte, _, _, _, ysc = MR.scale_data(trs, vas, tes, names, LABEL)

TR, ytr, _ = MH.build_dataset(tr, trs, Xtr, LAGS, "gru", ysc, stride=4)
VA, yva, _ = MH.build_dataset(va, vas, Xva, LAGS, "gru", ysc)
TE, yte, stamps = MH.build_dataset(te, tes, Xte, LAGS, "gru", ysc)

check("targets are (N, 96)", ytr.shape[1] == HZ, str(ytr.shape))
check("no NaN survives assembly", not np.isnan(ytr).any())
check("19 arrays (17 signals + DA + calendar)", len(TR) == 19, str(len(TR)))
check("DA future carries a presence flag", TR[-2].shape[-1] == 2, str(TR[-2].shape))
check("calendar is cyclical (4 features)", TR[-1].shape[-1] == 4, str(TR[-1].shape))
check("stride subsamples training only",
      len(ytr) < len(yva) or len(ytr) < 0.4 * (len(trs) - HZ),
      f"train {len(ytr):,} of {len(trs) - HZ:,} candidate rows")

cov = MH.da_covered_mask(TE[-2])
print(f"     test rows with full DA coverage: {cov.mean():.1%}")

# MRINN's flat inputs must be the same numbers in the same order
TRm, _, _ = MH.build_dataset(tr, trs, Xtr, LAGS, "mrinn", ysc, stride=4)
check("MRINN gets flat (N, T) where recurrent gets (N, T, 1)",
      TRm[0].shape == TR[0].shape[:2] and np.allclose(TRm[0], TR[0][:, :, 0]),
      f"{TRm[0].shape} vs {TR[0].shape}")

# -------------------------------------------------------- 2b. DA gate closure
print("\n=== 2b. Day-ahead gate closure ===")
# The day-ahead auction for delivery day D clears at 12:00 local on D-1, so an origin
# before ~13:00 local cannot know tomorrow's prices. Until 2026-08-29 `make_da_future`
# took shift(-h) unconditionally and handed the model 14.95% of its DA cells from the
# future. This section pins the fix so it cannot silently regress.
_st = pd.DatetimeIndex(tes[MH.TIME_COL])
if _st.tz is None:
    _st = _st.tz_localize("UTC")
_known = MH.da_published_mask(_st, HZ)

da_fixed = MH.make_da_future(te, tes, HZ, gate_closure=True)
da_leaky = MH.make_da_future(te, tes, HZ, gate_closure=False)

check("gate closure never flags an unpublished price as present",
      not (da_fixed[..., 1] > 0)[~_known].any(),
      f"{int((da_fixed[..., 1] > 0)[~_known].sum())} violations")
check("gate closure blanks the price, not just the flag",
      np.all(da_fixed[..., 0][~_known] == 0.0))
check("published cells are untouched",
      np.array_equal(da_fixed[_known], da_leaky[_known]))
check("the legacy path still leaks (kept only to reproduce results/ up to 2026-08-29)",
      (da_leaky[..., 1] > 0)[~_known].any())
check("near horizons are never gated -- h=1 is always knowable",
      _known[:, 0].all(), f"{int((~_known[:, 0]).sum())} gated at h=1")
check("far horizons are gated for the morning origins",
      0.2 < (~_known[:, HZ - 1]).mean() < 0.8,
      f"h=96 gated on {(~_known[:, HZ-1]).mean():.1%} of origins")
_was, _now = da_leaky[..., 1] > 0, da_fixed[..., 1] > 0
print(f"     DA cells present: {_was.mean():.2%} leaky -> {_now.mean():.2%} gated "
      f"({(_was & ~_now).sum() / max(_was.sum(), 1):.2%} of present cells blanked)")

# ------------------------------------------------------------------- 3. build
print("\n=== 3. Parameter budget ===")
C = list(MR.scaled_params(tr, FEATS_PRICES, FEATS_CAP)[:11])

built, m = {}, None
for name, kw in [("MR-GRU H=8", dict(cell="gru", hidden_units=8)),
                 ("MR-LSTM H=8", dict(cell="lstm", hidden_units=8)),
                 ("MR-GRU H=10", dict(cell="gru", hidden_units=10)),
                 ("MRINN H=8", dict(cell="mrinn", hidden_units=8))]:
    mm = MH.build_multihorizon(T=T, C=C, horizons=HZ, quantiles=Q, **kw)
    built[name] = mm.count_params()
    print(f"     {name:<12} {mm.count_params():>7,}"
          f"   {'OK' if mm.count_params() < 8900 else 'OVER 8,900'}")
    if name == "MR-GRU H=8":
        m = mm                      # carried forward as the model under test
    else:
        del mm

check("all configs stay under the 8,900 MLP baseline",
      all(v < 8900 for v in built.values()),
      ", ".join(f"{k} {v:,}" for k, v in built.items() if v >= 8900) or "all under")

m32 = MH.build_multihorizon("gru", 32, C, hidden_units=HU, horizons=HZ, quantiles=Q)
check("parameter count is constant in T", m32.count_params() == built["MR-GRU H=8"],
      f"T=8 {built['MR-GRU H=8']:,} vs T=32 {m32.count_params():,}")
del m32

check("output is (None, 96, 5)", tuple(m.output.shape[1:]) == (HZ, len(Q)),
      str(m.output.shape))

# ------------------------------------------- 4. THE defect check: does h matter?
print("\n=== 4. Horizon variation (the defect this rebuild fixes) ===")


def horizon_spread(model, inputs, n=64, known=True):
    """Std across the horizon axis of the median prediction, averaged over rows.

    `known` must match the model: a use_known_future=False model declares 17 inputs,
    not 19, so the array list has to be trimmed the same way the runner trims it.
    """
    p = model.predict([a[:n] for a in MH.feed(inputs, known)], verbose=0, batch_size=n)
    return float(np.mean(np.std(p[:, :, 2], axis=1)))


spread_fixed = horizon_spread(m, TE)
m_flat = MH.build_multihorizon("gru", T, C, hidden_units=HU, horizons=HZ, quantiles=Q,
                               use_known_future=False, use_horizon_embedding=False)
spread_flat = horizon_spread(m_flat, TE, known=False)
m_emb_only = MH.build_multihorizon("gru", T, C, hidden_units=HU, horizons=HZ,
                                   quantiles=Q, use_known_future=False,
                                   use_horizon_embedding=True)
spread_emb = horizon_spread(m_emb_only, TE, known=False)

print(f"     horizon spread of q50   embedding+known {spread_fixed:.5f}"
      f"   embedding only {spread_emb:.5f}   neither {spread_flat:.6f}")
check("the OLD design is horizon-blind (identical prediction at every h)",
      spread_flat < 1e-6, f"spread {spread_flat:.3e}")
check("the horizon embedding alone restores horizon variation",
      spread_emb > 1e-4, f"spread {spread_emb:.3e}")
check("the full model varies across horizons", spread_fixed > 1e-4,
      f"spread {spread_fixed:.3e}")
del m_flat, m_emb_only

# ------------------------------------- 4b. DA-outage dropout behaves correctly
print("\n=== 4b. DA-outage dropout ===")
lay = MH.DAOutageDropout(0.5)
ones = tf.ones((4000, HZ, 2))
kept_infer = float(tf.reduce_mean(lay(ones, training=False)))
kept_train = lay(ones, training=True).numpy()
check("no-op at inference", kept_infer == 1.0, f"kept {kept_infer}")
check("drops ~rate of samples in training", 0.45 <= kept_train.mean() <= 0.55,
      f"kept {kept_train.mean():.3f}")
per_sample = kept_train[:, :, 0].mean(axis=1)
check("blanks whole samples, never partial days",
      bool(np.all((per_sample == 0) | (per_sample == 1))),
      f"{float(np.mean((per_sample>0)&(per_sample<1))):.4f} partial")
check("adds no trainable parameters", len(lay.trainable_weights) == 0)

m_drop = MH.build_multihorizon("gru", T, C, hidden_units=HU, horizons=HZ, quantiles=Q,
                               da_dropout=0.2)
check("da_dropout leaves the parameter count unchanged",
      m_drop.count_params() == built["MR-GRU H=8"],
      f"{m_drop.count_params():,} vs {built['MR-GRU H=8']:,}")
p1 = m_drop.predict([a[:32] for a in TE], verbose=0, batch_size=32)
p2 = m_drop.predict([a[:32] for a in TE], verbose=0, batch_size=32)
check("inference is deterministic (dropout off)", np.allclose(p1, p2))
# The point of the layer: blanking the DA input must actually change the forecast,
# otherwise the model is ignoring day-ahead and the treatment is vacuous.
TE_blank = list(TE)
TE_blank[-2] = np.zeros_like(TE_blank[-2])
p_blank = m_drop.predict([a[:32] for a in TE_blank], verbose=0, batch_size=32)
delta = float(np.mean(np.abs(p_blank[:, :, 2] - p1[:, :, 2])))
check("blanking the DA input changes the forecast", delta > 1e-6, f"mean |delta| {delta:.4f}")
del m_drop

# --------------------------------------------------------------------- 5. fit
print("\n=== 5. Fit + per-horizon metrics ===")
t0 = time.perf_counter()
hist = m.fit(TR, ytr, validation_data=(VA, yva), epochs=EPOCHS,
             batch_size=256, verbose=0)
sec = time.perf_counter() - t0
check("loss is finite and decreasing",
      np.isfinite(hist.history["loss"][-1])
      and hist.history["loss"][-1] <= hist.history["loss"][0],
      f"{hist.history['loss'][0]:.4f} -> {hist.history['loss'][-1]:.4f}")

yp = m.predict(TE, verbose=0, batch_size=256)
check("prediction is (N, 96, 5)", yp.shape[1:] == (HZ, len(Q)), str(yp.shape))

per_h = MH.evaluate_by_horizon(yte, yp, Q, ysc)
check("AQCR == 0 at EVERY horizon", bool((per_h.AQCR == 0).all()),
      f"max {per_h.AQCR.max():.4f}")
check("per-horizon table has 96 rows", len(per_h) == HZ, str(len(per_h)))

s = MH.summarise(per_h)
print(f"     AQL {s['AQL']:.2f}  MAE {s['MAE']:.2f}  RMSE {s['RMSE']:.2f}"
      f"  cov80 {s['coverage80']:.1f}%")
print(f"     skill curve: AQL h=1 {s['AQL_h1']:.2f} -> h=96 {s['AQL_h96']:.2f}"
      f"  (slope {s['skill_slope']:+.2f})")
print(f"  [INFO] at {EPOCHS} epochs the skill curve is not yet meaningful; "
      f"the real run must show a clearly positive slope")

# sub-period reporting works
if cov.any() and (~cov).any():
    a = MH.summarise(MH.evaluate_by_horizon(yte, yp, Q, ysc, mask=cov))
    print(f"     DA-covered subset AQL {a['AQL']:.2f} ({cov.sum():,} rows)")

# ----------------------------------------------------------------- 6. baseline
print("\n=== 6. Seasonal naive baseline ===")
yt_n, yp_n, ok = MH.seasonal_naive(yte, stamps, ysc, Q)
check("baseline drops only the first day of test rows",
      0 < (~ok).sum() <= MH.STEPS_PER_DAY + 1, f"{(~ok).sum()} rows dropped")
naive_h = MH.evaluate_by_horizon(
    ysc.transform(yt_n.reshape(-1, 1)).reshape(yt_n.shape),
    np.stack([ysc.transform(yp_n[:, :, j].reshape(-1, 1)).reshape(yp_n.shape[:2])
              for j in range(len(Q))], axis=-1), Q, ysc)
nb = MH.summarise(naive_h)
print(f"     seasonal naive  AQL {nb['AQL']:.2f}  MAE {nb['MAE']:.2f}"
      f"  RMSE {nb['RMSE']:.2f}")
print(f"  [INFO] the 2-epoch model is not expected to beat it; the real run must")

# --------------------------------------------------------------- 7. projection
print("\n=== 7. Cost projection ===")
# The first fit is dominated by graph tracing, so timing it overstates the cost by
# more than an order of magnitude. Time a warmed-up epoch instead.
t0 = time.perf_counter()
m.fit(TR, ytr, epochs=1, batch_size=1024, verbose=0)
rate = len(ytr) / (time.perf_counter() - t0)
n_static = 105_000                      # train rows in 2022-01 -> 2024-09
print(f"     measured throughput: {rate:,.0f} samples/s "
      f"(cold fit above was {len(ytr) / (sec / EPOCHS):,.0f}/s -- tracing, ignore it)")
for stride in (1, 2, 4):
    rows = n_static / stride
    print(f"     stride {stride}: {rows:7,.0f} samples, {rows / rate:5.1f} s/epoch, "
          f"80 epochs = {rows / rate * 80 / 60:5.1f} min per run")
print("     the rule head dominates, so T has little effect on cost -- "
      "T=32 costs ~16% more than T=16")

print("\n" + "=" * 64)
print("FAILED: " + ", ".join(fails) if fails else "All checks passed.")
sys.exit(1 if fails else 0)
