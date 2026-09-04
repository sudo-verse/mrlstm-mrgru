"""Multi-horizon day-ahead forecasting: price at t+1 ... t+96 in one forward pass.

At origin tau (the last interval with complete observed data) this predicts the
imbalance price for every 15-minute interval out to 24 hours. Horizon h=1 is the
interval being settled now; h=96 is the same time tomorrow.

Rolling a one-step model forward is not an option: it would need forecasts of all
17 input signals, which is a strictly harder problem than the one being solved.
So the target is a vector and all 96 horizons are produced together.

WHAT THIS FIXES
---------------
`dayahead.py` broadcasts each signal's encoder output across the horizon axis with
RepeatVector and nothing else varies with h except the day-ahead price and the
calendar. The consequence is severe: with known-future covariates ablated, every
horizon receives *byte-identical* input and the model cannot express any horizon
variation at all. Even with them on, recent history contributes the same value to
h=1 and h=96, so the model cannot be sharper at 15 minutes than at 24 hours -- and
the measured skill curve was indeed flat (AQL 34.5 at h=1 vs 34.9 at h=96) where it
should rise steeply.

The fix is a learned `HorizonEmbedding`: one vector per horizon, concatenated onto
every signal's repeated representation and mixed by a Dense layer that is SHARED
across all 17 signals. Shared, not per-signal, is what keeps the model under the
8,900-parameter MLP baseline (104 params instead of 17 x 104), which is one of the
project's load-bearing claims.

Everything downstream -- the balancing-energy, market-reference and scarcity rules,
the smooth min/max combination and the hierarchical quantile head -- is imported
from the MRINN checkout UNCHANGED. The horizon axis is folded into the batch axis
before the rule head and unfolded after, so rule functions written for (N, H) apply
at (N*96, H) with weights shared across horizons, which is what "the same rulebook
applies at every horizon" ought to mean.

TRAINING REGIMES
----------------
Three, and the choice matters more than the encoder does:

  `expanding_folds()`  the MRINN paper's own protocol (Yu et al. 2026, Table 2) --
                       3 folds, training from 2022-01 up to 4 months before each test
                       block. The ONLY regime whose numbers are comparable to the
                       paper's Table 3.
  `rolling_folds()`    trailing 12 months, retrained per quarter. Drops the 2022
                       crisis regime (mean 235 EUR/MWh against 73-97 for 2023-25).
  `static_split()`     one fit ending 2024-09. Kept as a control only: it predicts
                       December 2025 from data 15 months stale, which no published
                       protocol does, and it consequently flatters everything it is
                       compared against.

THE DAY-AHEAD OUTAGE
--------------------
`P_DA_nemo` is exactly zero for the final 6,141 intervals (from 2025-10-28). The
presence flag was meant to handle this and does not: every static model that consumed
the day-ahead price scored BELOW seasonal naive on those rows (MR-GRU -7.2%, MRINN
-14.6%, MR-LSTM -19.6%) even though Nov-Dec is the calmest stretch of the year. With
no outage anywhere in training the flag is a constant 1, so the flag=0 branch is never
exercised. `DAOutageDropout` trains that branch. See its docstring.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from . import _mrinn_path  # noqa: F401  (side effect: sys.path)
from .seqdata import FEATURE_BASES, INPUT_NAMES

from library_imbalance.model import (  # noqa: E402
    HierarchicalQuantileHeadQ50,
    get_P_EX,
    get_P_RE,
    get_P_SC,
    smooth_max,
    smooth_min,
)

TIME_COL = "Time [UTC] start"
STEPS_PER_DAY = 96          # 15-minute intervals in a day
DA_COL = "P_DA_nemo"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

_CELLS = {"lstm": layers.LSTM, "gru": layers.GRU}


# ===================================================================== splits

def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _fmt(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S%z")[:-2] + ":00"


def static_split(train_start="2022-01-01", val_start="2024-09-01",
                 test_start="2025-01-01", test_end="2026-01-01") -> Dict[str, tuple]:
    """One fixed split: train on everything since 2022, test the last 12 months."""
    a, b, c, d = (_ts(x) for x in (train_start, val_start, test_start, test_end))
    return {"tag": "static",
            "train": (_fmt(a), _fmt(b)),
            "val": (_fmt(b), _fmt(c)),
            "test": (_fmt(c), _fmt(d))}


def rolling_folds(test_start="2025-01-01", test_end="2026-01-01",
                  test_months=3, train_months=12, val_months=3) -> List[Dict[str, tuple]]:
    """Walk-forward folds with a trailing fixed-length training window.

    For each test quarter the model is retrained on the `train_months` immediately
    preceding the validation block, which itself immediately precedes the test
    block. Nothing at or after the test start is ever seen in training.

    This is the "1-year sliding window" idea done properly. It is worth testing
    here because 2022 is a different price regime (mean 235 EUR/MWh against 73-97
    for 2023-2025), so a statically-trained model spends a third of its data
    fitting a market that no longer exists.
    """
    folds, start, stop = [], _ts(test_start), _ts(test_end)
    i = 0
    while start < stop:
        te_end = min(start + pd.DateOffset(months=test_months), stop)
        va_start = start - pd.DateOffset(months=val_months)
        tr_start = va_start - pd.DateOffset(months=train_months)
        folds.append({"tag": f"roll{i}",
                      "train": (_fmt(tr_start), _fmt(va_start)),
                      "val": (_fmt(va_start), _fmt(start)),
                      "test": (_fmt(start), _fmt(te_end))})
        start, i = te_end, i + 1
    return folds


def expanding_folds(train_start="2022-01-01", test_start="2025-01-01",
                    test_end="2026-01-01", test_months=4,
                    val_months=4) -> List[Dict[str, tuple]]:
    """The MRINN paper's own evaluation protocol (Yu et al. 2026, Table 2).

    Three folds, each retrained on everything from `train_start` up to the validation
    block, which immediately precedes the test block:

        fold 1   train 2022-01 -> 2024-09   val 2024-09 -> 2025-01   test 2025-01 -> 2025-05
        fold 2   train 2022-01 -> 2025-01   val 2025-01 -> 2025-05   test 2025-05 -> 2025-09
        fold 3   train 2022-01 -> 2025-05   val 2025-05 -> 2025-09   test 2025-09 -> 2026-01

    (The paper writes the boundaries as 2024-08/2025-04/2025-08 inclusive; these are
    the same cuts expressed as half-open intervals.)

    This is neither `static_split` nor `rolling_folds`. The training window *expands*
    rather than sliding, so it keeps the 2022 energy-crisis regime while still seeing
    data up to four months before each test block. It exists here because it is the
    only regime whose numbers are directly comparable to the paper's Table 3 --
    `static_split` trains to 2024-09 and then predicts December 2025 fifteen months
    later, which no published protocol does and which flatters the alternatives.
    """
    folds, start, stop = [], _ts(test_start), _ts(test_end)
    tr_start, i = _ts(train_start), 0
    while start < stop:
        te_end = min(start + pd.DateOffset(months=test_months), stop)
        va_start = start - pd.DateOffset(months=val_months)
        folds.append({"tag": f"exp{i}",
                      "train": (_fmt(tr_start), _fmt(va_start)),
                      "val": (_fmt(va_start), _fmt(start)),
                      "test": (_fmt(start), _fmt(te_end))})
        start, i = te_end, i + 1
    return folds


# ======================================================================= data

def make_multihorizon_targets(frame: pd.DataFrame, target_col: str,
                              horizons: int = STEPS_PER_DAY) -> np.ndarray:
    """(N, horizons) -- y[:, h-1] is the price h steps ahead. NaN where unavailable."""
    s = frame[target_col]
    return np.column_stack([s.shift(-h).to_numpy(dtype="float32")
                            for h in range(1, horizons + 1)])


def make_calendar(frame: pd.DataFrame, horizons: int = STEPS_PER_DAY) -> np.ndarray:
    """(N, horizons, 4) -- sin/cos of time-of-day and day-of-week at t+h.

    Cyclical rather than raw integers, so 23:45 and 00:00 are adjacent rather than
    maximally distant. At a 24-hour horizon the daily and weekly cycles carry more
    signal than anything in recent history, and they are known exactly.
    """
    idx = pd.DatetimeIndex(frame[TIME_COL])
    cal = np.empty((len(frame), horizons, 4), dtype="float32")
    for h in range(1, horizons + 1):
        t = idx + pd.Timedelta(minutes=15 * h)
        tod = (t.hour * 60 + t.minute).to_numpy() / 1440.0
        dow = t.dayofweek.to_numpy() / 7.0
        cal[:, h - 1, 0] = np.sin(2 * np.pi * tod)
        cal[:, h - 1, 1] = np.cos(2 * np.pi * tod)
        cal[:, h - 1, 2] = np.sin(2 * np.pi * dow)
        cal[:, h - 1, 3] = np.cos(2 * np.pi * dow)
    return cal


MARKET_TZ = "Europe/Vienna"     # delivery-day calendar of the day-ahead auction
PUBLISH_LOCAL_HOUR = 13         # SDAC results are public ~12:55 local on D-1


def da_published_mask(stamps: pd.DatetimeIndex, horizons: int = STEPS_PER_DAY) -> np.ndarray:
    """(N, horizons) bool -- was the day-ahead price for t+h public at origin t?

    The day-ahead auction for delivery day D clears at 12:00 local on D-1 and results
    are published shortly after. So at an origin before ~13:00 local, tomorrow's prices
    do not exist yet, and any horizon reaching into tomorrow is unknowable.

    Without this, `make_da_future` hands the model 14.95% of its day-ahead cells from
    the future: every origin between 00:00 and 12:45 local (54.17% of all origins) and
    every horizon from h=45 upward, rising to 54.17% of origins at h=96. See
    `audit_gate_closure.py` for the measurement.
    """
    # Day arithmetic on naive wall-clock, then re-localize: adding a Timedelta to a
    # tz-aware index is absolute, which lands an hour off on the two DST days.
    local_naive = stamps.tz_convert(MARKET_TZ).tz_localize(None)
    pub_naive = (local_naive.normalize() - pd.Timedelta(days=1)
                 + pd.Timedelta(hours=PUBLISH_LOCAL_HOUR))
    pub = (pd.DatetimeIndex(pub_naive)
           .tz_localize(MARKET_TZ, nonexistent="shift_forward", ambiguous=True)
           .tz_convert("UTC").to_numpy())
    origin = stamps.to_numpy()

    n = len(stamps)
    known = np.ones((n, horizons), dtype=bool)
    for h in range(1, horizons + 1):
        tgt = np.arange(n) + h
        ok = tgt < n
        known[ok, h - 1] = pub[tgt[ok]] <= origin[ok]
    return known


def make_da_future(raw_frame: pd.DataFrame, shifted_frame: pd.DataFrame,
                   horizons: int = STEPS_PER_DAY, da_col: str = DA_COL,
                   gate_closure: bool = True) -> np.ndarray:
    """(N, horizons, 2) -- day-ahead price at t+h, plus an 'is published' flag.

    The day-ahead price is the one input genuinely known a day in advance, so it is
    read from the raw split (shift_data keeps only lag columns) and aligned onto the
    shifted frame's timestamps.

    The presence flag is not cosmetic. P_DA_nemo is zero-filled wherever no price was
    published -- including every row from 2025-10-28 to the end of the record, 6,141
    intervals -- and a zero price is not the same as a missing one. Without the flag
    the model would learn that day-ahead is sometimes exactly zero.

    `gate_closure=True` additionally blanks prices that had not yet been auctioned at
    the origin instant (see `da_published_mask`). This is the honest setting and is the
    default. `gate_closure=False` reproduces the leaky behaviour that produced every
    result in `results/` up to 2026-08-29 -- keep it only for reproducing those.
    """
    s_da = raw_frame[da_col]
    lead = np.column_stack([s_da.shift(-h).to_numpy(dtype="float32")
                            for h in range(1, horizons + 1)])

    pos = pd.Series(np.arange(len(raw_frame)),
                    index=pd.DatetimeIndex(raw_frame[TIME_COL]))
    take = pos.reindex(pd.DatetimeIndex(shifted_frame[TIME_COL])).to_numpy()
    if np.isnan(take).any():
        raise ValueError("shifted frame contains timestamps absent from the raw split")
    lead = lead[take.astype(int)]

    present = np.where(np.isnan(lead), 0.0, (lead != 0.0)).astype("float32")
    lead = np.nan_to_num(lead, nan=0.0)

    if gate_closure:
        stamps = pd.DatetimeIndex(shifted_frame[TIME_COL])
        if stamps.tz is None:
            stamps = stamps.tz_localize("UTC")
        known = da_published_mask(stamps, horizons)
        # Blank the price as well as the flag: an unpublished price must reach the
        # model as "missing", which is exactly the state DAOutageDropout trains for.
        present = present * known
        lead = lead * known

    return np.stack([lead, present], axis=-1)


def to_inputs(frame: pd.DataFrame, lags: Sequence[int], kind: str) -> List[np.ndarray]:
    """The 17 history inputs. MRINN wants flat (N, T); recurrent models want (N, T, 1).

    Column `{base}_lag1` is the most recent step, so the lag axis is reversed to put
    the oldest timestep first -- which is the order a recurrent cell must read.
    """
    lags = sorted(int(L) for L in lags)
    out = []
    for base in FEATURE_BASES:
        cols = [f"{base}_lag{L}" for L in lags]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"missing lag columns for '{base}': {missing}")
        a = frame[cols].to_numpy(dtype="float32")            # newest -> oldest
        a = np.ascontiguousarray(a[:, ::-1])                 # oldest -> newest
        out.append(a if kind == "mrinn" else a[:, :, None])
    return out


def build_dataset(raw_frame, shifted_frame, scaled_frame, lags, kind, y_scaler,
                  target_col="imbalance_price", horizons=STEPS_PER_DAY, stride=1,
                  gate_closure=True):
    """Assemble (inputs, targets, timestamps) for one split.

    The final `horizons` rows are dropped: their future targets do not exist.
    `stride` subsamples rows and must be 1 for validation and test -- adjacent
    samples share 95 of their 96 targets, so dense sampling is largely redundant
    compute during training but throwing away test rows would just be throwing away
    evaluation.
    """
    tgt = make_multihorizon_targets(shifted_frame, target_col, horizons)
    da = make_da_future(raw_frame, shifted_frame, horizons, gate_closure=gate_closure)
    cal = make_calendar(shifted_frame, horizons)
    hist = to_inputs(scaled_frame, lags, kind)
    stamps = pd.DatetimeIndex(shifted_frame[TIME_COL]).to_numpy()

    keep = slice(None, -horizons)
    hist = [h[keep] for h in hist]
    da, cal, tgt, stamps = da[keep], cal[keep], tgt[keep], stamps[keep]

    good = ~np.isnan(tgt).any(axis=1)
    if stride > 1:
        sel = np.zeros(len(tgt), dtype=bool)
        sel[::stride] = True
        good &= sel
    hist = [h[good] for h in hist]
    da, cal, tgt, stamps = da[good], cal[good], tgt[good], stamps[good]

    tgt = y_scaler.transform(tgt.reshape(-1, 1)).reshape(tgt.shape).astype("float32")
    return hist + [da, cal], tgt, stamps


# ======================================================================= loss

def multihorizon_pinball_loss(quantiles: Sequence[float]):
    """Pinball loss averaged over horizons and quantiles.

    y_true (B, Hz)   y_pred (B, Hz, Q)

    Weighting is uniform across horizons, which is what "forecast the whole day"
    means -- but it does mean h=1 receives roughly 1/96 of the gradient, so this
    model will be worse at 15 minutes than a dedicated one-step model. That cost is
    structural, and `evaluate_by_horizon` makes it visible rather than hiding it.
    """
    qs = tf.reshape(tf.constant(list(quantiles), tf.float32), (1, 1, -1))

    def loss(y_true, y_pred):
        y_true = tf.expand_dims(tf.cast(y_true, tf.float32), -1)
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(qs * e, (qs - 1.0) * e))
    return loss


# ====================================================================== model

@tf.keras.utils.register_keras_serializable(package="mrrnn")
class HorizonEmbedding(layers.Layer):
    """One learned vector per forecast horizon, tiled across the batch.

    This is the whole fix. Without it the rule head sees identical inputs at h=1 and
    h=96 and the model is structurally incapable of being sharper at short horizons.
    Costs `horizons * dim` parameters (96 x 4 = 384) and nothing else.
    """

    def __init__(self, horizons: int, dim: int = 4, **kw):
        super().__init__(**kw)
        self.horizons, self.dim = int(horizons), int(dim)

    def build(self, input_shape):
        self.table = self.add_weight(
            shape=(self.horizons, self.dim), initializer="glorot_uniform",
            trainable=True, name="table")
        super().build(input_shape)

    def call(self, ref):
        # `ref` supplies the batch size only; its values are never read.
        batch = tf.shape(ref)[0]
        return tf.tile(tf.expand_dims(self.table, 0), [batch, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.horizons, self.dim)

    def get_config(self):
        return {**super().get_config(), "horizons": self.horizons, "dim": self.dim}


@tf.keras.utils.register_keras_serializable(package="mrrnn")
class DAOutageDropout(layers.Layer):
    """Randomly simulate a day-ahead publication outage during training.

    Input/output is the (batch, horizons, 2) day-ahead tensor [price, is_published].
    With probability `rate` an entire sample is blanked -- price to 0 AND flag to 0 --
    which is exactly the state the real feed entered on 2025-10-28 and held for the
    last 6,141 intervals of the record.

    Why a whole sample rather than scattered horizons: a publication failure takes out
    the entire delivery day at once, so per-horizon masking would train the model to
    interpolate around gaps it will never actually see.

    This exists because the presence flag alone does not work. Every static model that
    consumed the day-ahead price fell BELOW the seasonal-naive baseline in Q4 2025
    (MR-GRU -7.2%, MRINN -14.6%, MR-LSTM -19.6%) despite Nov-Dec being the calmest
    stretch of the year. With no outage anywhere in training the flag is a constant 1,
    the flag=0 branch is never exercised, and at test time a hard zero arrives at a
    weight tuned for ~100 EUR/MWh. Zero trainable parameters; it only changes what the
    existing weights are asked to survive.
    """

    def __init__(self, rate: float = 0.2, **kw):
        super().__init__(**kw)
        self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate <= 0.0:
            return x
        keep = tf.cast(
            tf.random.uniform((tf.shape(x)[0], 1, 1)) >= self.rate, x.dtype)
        return x * keep

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


def _encode(x, kind: str, hidden_units: int, num_layer: int, name: str):
    """Per-signal encoder. This single branch is the entire model contribution."""
    if kind == "mrinn":
        for i in range(num_layer):
            x = layers.Dense(hidden_units, activation="swish",
                             name=f"{name}_dense_{i}")(x)
        return x
    Cell = _CELLS[kind]
    for i in range(num_layer):
        x = Cell(hidden_units, return_sequences=(i < num_layer - 1),
                 name=f"{name}_{kind}_{i}")(x)
    return x


def feed(arrays, use_known_future: bool):
    """Trim the assembled array list to the inputs the model actually declares."""
    return arrays if use_known_future else arrays[:len(INPUT_NAMES)]


def build_multihorizon(cell: str, T: int, C: Sequence[float], hidden_units: int = 8,
                       num_layer: Optional[int] = None, horizons: int = STEPS_PER_DAY,
                       quantiles: Sequence[float] = QUANTILES,
                       use_known_future: bool = True,
                       use_horizon_embedding: bool = True,
                       horizon_embed_dim: int = 4, lr: float = 1e-3,
                       da_dropout: float = 0.0):
    """MR-GRU / MR-LSTM / MRINN predicting `horizons` steps ahead in one shot.

    C is the ordered list C0..C10 of scaled regulatory constants.

    da_dropout > 0 blanks the whole day-ahead input on a random `da_dropout` fraction
    of training samples, so the model learns a fallback for the outage the presence
    flag alone does not survive. Ignored when use_known_future=False (there is nothing
    left to blank). See `DAOutageDropout`.

    use_horizon_embedding=False reproduces the defect in `dayahead.py` and is kept
    solely as the control that demonstrates it: with it off AND known-future off,
    the model emits an identical prediction at every horizon.

    use_known_future=False ablates the day-ahead price and calendar, leaving a purely
    history-driven model -- the control that says what those covariates are worth at
    a 24-hour horizon.
    """
    cell = cell.lower()
    if cell not in _CELLS and cell != "mrinn":
        raise ValueError(f"cell must be 'mrinn' or one of {sorted(_CELLS)}")
    quantiles = list(quantiles)
    if num_layer is None:
        num_layer = 2 if cell == "mrinn" else 1
    Hu, Hz, Q = hidden_units, horizons, len(quantiles)

    shape = (T,) if cell == "mrinn" else (T, 1)
    hist_in = [layers.Input(shape=shape, name=n) for n in INPUT_NAMES]
    da_in = layers.Input(shape=(Hz, 2), name="da_future_in")
    cal_in = layers.Input(shape=(Hz, 4), name="calendar_in")

    # --- history: one encoder per signal, then broadcast over the horizon axis ---
    reps = [_encode(x, cell, Hu, num_layer, base)
            for x, base in zip(hist_in, FEATURE_BASES)]
    reps = [layers.RepeatVector(Hz, name=f"rep_{b}")(r)
            for r, b in zip(reps, FEATURE_BASES)]

    # --- make the history representation horizon-aware (the fix) ---
    if use_horizon_embedding:
        emb = HorizonEmbedding(Hz, horizon_embed_dim,
                               name="horizon_embedding")(reps[0])
        mix = layers.Dense(Hu, activation="swish", name="horizon_mix")  # SHARED
        reps = [mix(layers.Concatenate(name=f"hcat_{b}")([r, emb]))
                for r, b in zip(reps, FEATURE_BASES)]

    (V, E_ap, E_mp, P_ap, P_mp, E_an, E_mn, P_an, P_mn,
     P_vp, P_vn, P15, P60, PDA, L15, L60, LDA) = reps

    known_rep = None
    if use_known_future:
        # Blank before BOTH consumers, so an outage is consistent across the two paths
        # rather than the model recovering the price it was meant to have lost.
        da_x = (DAOutageDropout(da_dropout, name="da_outage")(da_in)
                if da_dropout > 0.0 else da_in)
        # The rule needs the day-ahead price AT THE TARGET INTERVAL, which is known
        # a day in advance -- not one inferred from history. Substitute it.
        PDA = layers.Dense(Hu, activation="swish", name="da_future_rep")(da_x)
        known_rep = layers.Dense(Hu, activation="swish", name="known_future_rep")(
            layers.Concatenate(name="known_future_cat")([da_x, cal_in]))

    # --- fold horizons into the batch axis so MRINN's rule blocks apply unchanged ---
    fold = layers.Lambda(lambda t: tf.reshape(t, (-1, Hu)), name="fold")
    (Vf, E_apf, E_mpf, P_apf, P_mpf, E_anf, E_mnf, P_anf, P_mnf,
     P_vpf, P_vnf, P15f, P60f, PDAf, L15f, L60f, LDAf) = [
        fold(t) for t in (V, E_ap, E_mp, P_ap, P_mp, E_an, E_mn, P_an, P_mn,
                          P_vp, P_vn, P15, P60, PDA, L15, L60, LDA)]

    # ---------------- imported from MRINN, unchanged ----------------
    P_RE = get_P_RE(E_apf, E_mpf, E_anf, E_mnf, Vf,
                    P_apf, P_mpf, P_anf, P_mnf, P_vpf, P_vnf, Hu)
    P_EX, P_base = get_P_EX(P15f, L15f, P60f, L60f, PDAf, Vf, *C[:7])
    P_SC = get_P_SC(P_base, Vf, *C[7:11])

    lo = smooth_min(smooth_min(P_RE, P_EX), P_SC)
    hi = smooth_max(smooth_max(P_RE, P_EX), P_SC)
    W = layers.Dense(2, activation="softmax", name="final_gate")(Vf)
    z = layers.Add(name="imbalance_price_rep")([
        layers.Multiply()([lo, W[:, 0:1]]),
        layers.Multiply()([hi, W[:, 1:2]]),
    ])
    # ---------------------------------------------------------------

    if use_known_future:
        # w_DA averages 0.003 in this market -- P_EX all but ignores the day-ahead
        # price -- so without a direct path the strongest known-future signal is
        # discarded by the rule itself. Ablate with use_known_future=False.
        z = layers.Dense(Hu, activation="swish", name="rep_with_known")(
            layers.Concatenate(name="rep_concat")([z, fold(known_rep)]))

    out = HierarchicalQuantileHeadQ50(z, quantiles, name_prefix="imbalance_price_hq")
    out = layers.Lambda(lambda t: tf.reshape(t, (-1, Hz, Q)), name="unfold")(out)

    tag = "MRINN" if cell == "mrinn" else f"MR{cell.upper()}"
    # Keras rejects an Input with no outbound node ("`inputs` not connected to
    # `outputs`"), so the known-future ABLATION is built without da_in/cal_in rather
    # than leaving them dangling. Keras 3.13 enforces this; 3.15 does not.
    model_in = hist_in + ([da_in, cal_in] if use_known_future else [])
    model = models.Model(model_in, out, name=f"{tag}_h{Hz}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=multihorizon_pinball_loss(quantiles))
    return model


# ================================================================= evaluation

def _inv(a, y_scaler):
    """Back to EUR/MWh. `y_scaler=None` means the values are already on that scale.

    The None path matters for walk-forward evaluation: every rolling fold fits its own
    RobustScaler on its own training window, so folds must be inverted individually and
    only then concatenated. Pooling scaled values and inverting once would apply the
    wrong median and IQR to every fold but the last.
    """
    a = np.asarray(a)
    if y_scaler is None:
        return a
    return y_scaler.inverse_transform(a.reshape(-1, 1)).ravel().reshape(a.shape)


def evaluate_by_horizon(y_true, y_pred, quantiles=QUANTILES, y_scaler=None,
                        mask=None) -> pd.DataFrame:
    """Metrics at each horizon, on the original price scale.

    y_true (N, Hz)   y_pred (N, Hz, Q).  `mask` selects a subset of rows, which is
    how the day-ahead blackout period is reported separately.
    """
    quantiles = list(quantiles)
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    yt = _inv(y_true, y_scaler)
    yp = np.stack([_inv(y_pred[:, :, j], y_scaler) for j in range(y_pred.shape[-1])],
                  axis=-1)

    mid = quantiles.index(0.5)
    rows = []
    for h in range(y_true.shape[1]):
        t, p = yt[:, h], yp[:, h, :]
        e = t - p[:, mid]
        ql = [float(np.mean(np.maximum(q * (t - p[:, j]), (q - 1) * (t - p[:, j]))))
              for j, q in enumerate(quantiles)]
        rows.append({
            "h": h + 1, "minutes_ahead": (h + 1) * 15,
            "AQL": float(np.mean(ql)),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "AQCR": float((p[:, :-1] > p[:, 1:]).mean() * 100),
            "coverage80": float(((t >= p[:, 0]) & (t <= p[:, -1])).mean() * 100),
            "AIW": float(np.mean(p[:, -1] - p[:, 0])),
        })
    return pd.DataFrame(rows)


def summarise(per_h: pd.DataFrame) -> Dict[str, float]:
    """Horizon-averaged headline metrics, plus the skill curve's slope."""
    return {
        "AQL": float(per_h.AQL.mean()),
        "MAE": float(per_h.MAE.mean()),
        "RMSE": float(per_h.RMSE.mean()),
        "AQCR": float(per_h.AQCR.max()),
        "coverage80": float(per_h.coverage80.mean()),
        "AIW": float(per_h.AIW.mean()),
        "AQL_h1": float(per_h.AQL.iloc[0]),
        "AQL_h96": float(per_h.AQL.iloc[-1]),
        "skill_slope": float(per_h.AQL.iloc[-1] - per_h.AQL.iloc[0]),
    }


def climatology(y_true_scaled, y_scaler, quantiles=QUANTILES, reference=None):
    """The unconditional baseline: the same quantiles at every horizon, forever.

    This is the reference the project was missing. Seasonal naive turns out to be a
    *weak* baseline here -- its RMSE of 549 is about sqrt(2) x the price standard
    deviation of 390, the signature of differencing two nearly independent draws, so
    it loses to simply predicting the mean. Beating it by 32% therefore overstates
    the result; the honest comparison is against a constant, which the models beat by
    roughly 12% on MAE while matching its RMSE almost exactly.

    `reference` is the sample the quantiles are read from and should be the TRAINING
    prices. Passing None reads them from the test set itself, which is not a forecast
    and is only useful as an upper bound on what any constant could achieve.
    """
    y = _inv(y_true_scaled, y_scaler)
    src = y.ravel() if reference is None else np.asarray(reference).ravel()
    src = src[~np.isnan(src)]
    qs = np.quantile(src, list(quantiles))
    pred = np.broadcast_to(qs, y.shape + (len(qs),)).copy()
    return y, pred


def seasonal_naive(y_true_scaled, stamps, y_scaler, quantiles=QUANTILES,
                   resid_quantiles=None):
    """Price at the same interval yesterday -- the right baseline 24 hours out.

    Persistence (the one-step baseline) is meaningless here: the price 24 hours ago
    is a far better guess than the price 15 minutes ago when forecasting a full day.
    Quantiles come from the empirical distribution of the day-over-day residual, so
    the baseline is probabilistic and directly AQL-comparable.

    Row i's forecast for horizon h is the observed price at (stamps[i] - 24h) + h,
    which is row (i - one day) at the same horizon. The lookup goes through the
    timestamps rather than a fixed row offset, so DST shifts and missing intervals
    cannot silently misalign the baseline.

    Returns (y_true_subset, y_pred_subset, mask) on the original price scale, where
    `mask` selects the rows that have a genuine "yesterday" -- apply it to the
    model's predictions too, so both are scored on identical rows.
    """
    y = _inv(y_true_scaled, y_scaler)                       # (N, Hz) real prices
    idx = pd.DatetimeIndex(stamps)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    prev = pos.reindex(idx - pd.Timedelta(days=1)).to_numpy()
    ok = ~pd.isna(prev)
    if not ok.any():
        raise ValueError("no test row has a matching interval 24 hours earlier")

    base = y[np.where(ok, np.nan_to_num(prev, nan=0.0), 0).astype(int)]
    if resid_quantiles is None:
        # In-sample: the interval width is fitted on the very rows being scored, which
        # flatters the baseline by ~0.065 AQL. Conservative for the project's claim,
        # but still not a forecast -- pass day-over-day residuals from the TRAINING
        # split instead. Kept as the default only so existing callers stay reproducible.
        qs = np.quantile((y[ok] - base[ok]).ravel(), list(quantiles))
    else:
        qs = np.asarray(resid_quantiles, dtype=float)
        if qs.shape != (len(quantiles),):
            raise ValueError(f"resid_quantiles must have {len(quantiles)} entries")
    pred = np.stack([base + q for q in qs], axis=-1)        # (N, Hz, Q)
    return y[ok], pred[ok], ok


def da_covered_mask(da_future: np.ndarray) -> np.ndarray:
    """True for rows whose whole 24-hour window has a published day-ahead price.

    Read straight off the presence flag built by `make_da_future`, so it needs no
    hardcoded blackout date and stays correct if the data is extended or revised.
    Use it to report the day-ahead blackout (all of Nov-Dec 2025) as its own
    sub-period rather than letting it silently drag the headline numbers down.
    """
    return da_future[:, :, 1].min(axis=1) > 0.5
