"""Rules-free counterparts to the MR-* models, and two non-neural references.

The point of this module is a 2 x 3 design. Every model below shares the data
pipeline, the horizon embedding, the known-future path, `DAOutageDropout` and the
hierarchical quantile head with `multihorizon.build_multihorizon`. The ONLY thing
that differs is what sits between the per-signal encoders and the head:

                    | market-rule block            | plain Dense mixer
    ----------------+------------------------------+-------------------
    Dense encoder   | MRINN                        | MLP
    LSTM encoder    | MR-LSTM                      | LSTM
    GRU encoder     | MR-GRU                       | GRU

Reading down a column isolates the encoder (the project's original claim); reading
across a row isolates the market-rule blocks (the question an examiner asks first,
and one the project could not previously answer).

`attnbilstm` is the paper's strongest recurrent baseline and has no MR- counterpart.
`lqr` is a genuinely linear quantile regression -- flatten every input, one affine map
to horizons x quantiles, pinball loss. It is the only model here WITHOUT the
hierarchical head, which is deliberate: it shows what quantile crossing looks like
when the head does not forbid it, and so demonstrates that AQCR = 0 is earned by that
head rather than assumed.
"""

from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .multihorizon import (QUANTILES, STEPS_PER_DAY, DAOutageDropout, FEATURE_BASES,
                           HorizonEmbedding, INPUT_NAMES, multihorizon_pinball_loss)
from ._mrinn_path import MRINN_ROOT  # noqa: F401  (keeps the import boundary explicit)

from library_imbalance.model import HierarchicalQuantileHeadQ50

_RECURRENT = {"lstm": layers.LSTM, "gru": layers.GRU}


def _encode_plain(x, kind: str, units: int, num_layer: int, base: str):
    """One signal's history -> (N, units). Mirrors multihorizon._encode exactly."""
    if kind == "mlp":
        h = x
        for i in range(num_layer):
            h = layers.Dense(units, activation="swish", name=f"{base}_d{i}")(h)
        return h
    if kind == "attnbilstm":
        h = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True), name=f"{base}_bilstm")(x)
        # Additive attention over time, collapsed to one vector per signal.
        score = layers.Dense(1, activation="tanh", name=f"{base}_att")(h)
        w = layers.Softmax(axis=1, name=f"{base}_attw")(score)
        h = layers.Multiply(name=f"{base}_attmul")([h, w])
        h = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{base}_attsum")(h)
        return layers.Dense(units, activation="swish", name=f"{base}_attproj")(h)
    cell = _RECURRENT[kind]
    h = x
    for i in range(num_layer - 1):
        h = cell(units, return_sequences=True, name=f"{base}_{kind}{i}")(h)
    return cell(units, name=f"{base}_{kind}")(h)


def build_baseline(kind: str, T: int, hidden_units: int = 8,
                   num_layer: Optional[int] = None, horizons: int = STEPS_PER_DAY,
                   quantiles: Sequence[float] = QUANTILES,
                   use_known_future: bool = True,
                   use_horizon_embedding: bool = True,
                   horizon_embed_dim: int = 4, lr: float = 1e-3,
                   da_dropout: float = 0.0, lqr_l2: float = 1e-3):
    """A rules-free multi-horizon quantile model with the MR-* pipeline around it.

    `kind` is one of 'mlp', 'lstm', 'gru', 'attnbilstm', 'lqr'. Note that 'lstm' here
    is NOT MR-LSTM: it is MR-LSTM with the market-rule block deleted and replaced by a
    Dense layer of the same width, which is the ablation the rule blocks have never
    been subjected to in this project.
    """
    kind = kind.lower()
    quantiles = list(quantiles)
    Hu, Hz, Q = hidden_units, horizons, len(quantiles)
    if num_layer is None:
        num_layer = 2 if kind in ("mlp", "lqr") else 1

    flat = kind in ("mlp", "lqr")
    shape = (T,) if flat else (T, 1)
    hist_in = [layers.Input(shape=shape, name=n) for n in INPUT_NAMES]
    da_in = layers.Input(shape=(Hz, 2), name="da_future_in")
    cal_in = layers.Input(shape=(Hz, 4), name="calendar_in")

    # ---- linear quantile regression: one affine map, no head, no embedding ----
    if kind == "lqr":
        parts = [layers.Flatten()(x) for x in hist_in]
        if use_known_future:
            da_x = (DAOutageDropout(da_dropout, name="da_outage")(da_in)
                    if da_dropout > 0.0 else da_in)
            parts += [layers.Flatten()(da_x), layers.Flatten()(cal_in)]
        z = layers.Concatenate(name="flat_all")(parts)
        # RIDGE, and it is not optional. Given the same 544 lag features as every other
        # model, a dense linear map is 1,120 inputs x 480 outputs = 538k weights fitted
        # on 17.5k rows. Unregularised it overfits the train/val period hard and scores
        # WORSE than climatology on the next quarter (AQL 61 against 29), which is a
        # broken baseline rather than a finding. The paper's LQR carried 14 parameters
        # because it was given a handful of features; keeping the feature set identical
        # here preserves the single-factor comparison, so the capacity has to be
        # controlled with a penalty instead.
        out = layers.Dense(Hz * Q, name="linear_quantiles",
                           kernel_regularizer=tf.keras.regularizers.L2(lqr_l2))(z)
        out = layers.Reshape((Hz, Q), name="unfold")(out)
        model_in = hist_in + ([da_in, cal_in] if use_known_future else [])
        model = models.Model(model_in, out, name=f"LQR_h{Hz}")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=multihorizon_pinball_loss(quantiles))
        return model

    # ---- everything else: identical to build_multihorizon up to the rule block ----
    reps = [_encode_plain(x, kind, Hu, num_layer, base)
            for x, base in zip(hist_in, FEATURE_BASES)]
    reps = [layers.RepeatVector(Hz, name=f"rep_{b}")(r)
            for r, b in zip(reps, FEATURE_BASES)]

    if use_horizon_embedding:
        emb = HorizonEmbedding(Hz, horizon_embed_dim, name="horizon_embedding")(reps[0])
        mix = layers.Dense(Hu, activation="swish", name="horizon_mix")   # SHARED
        reps = [mix(layers.Concatenate(name=f"hcat_{b}")([r, emb]))
                for r, b in zip(reps, FEATURE_BASES)]

    known_rep = None
    if use_known_future:
        da_x = (DAOutageDropout(da_dropout, name="da_outage")(da_in)
                if da_dropout > 0.0 else da_in)
        known_rep = layers.Dense(Hu, activation="swish", name="known_future_rep")(
            layers.Concatenate(name="known_future_cat")([da_x, cal_in]))

    # THE ONLY STRUCTURAL DIFFERENCE FROM build_multihorizon: where MRINN applies
    # get_P_RE / get_P_EX / get_P_SC and the smooth min-max gate, this mixes the 17
    # signal representations with one learned Dense layer and nothing else.
    merged = layers.Concatenate(name="signal_concat")(reps)          # (N, Hz, 17*Hu)
    fold = layers.Lambda(lambda t: tf.reshape(t, (-1, t.shape[-1])), name="fold")
    z = layers.Dense(Hu, activation="swish", name="mixer")(fold(merged))

    if use_known_future:
        foldk = layers.Lambda(lambda t: tf.reshape(t, (-1, Hu)), name="fold_known")
        z = layers.Dense(Hu, activation="swish", name="rep_with_known")(
            layers.Concatenate(name="rep_concat")([z, foldk(known_rep)]))

    out = HierarchicalQuantileHeadQ50(z, quantiles, name_prefix="imbalance_price_hq")
    out = layers.Lambda(lambda t: tf.reshape(t, (-1, Hz, Q)), name="unfold")(out)

    model_in = hist_in + ([da_in, cal_in] if use_known_future else [])
    model = models.Model(model_in, out, name=f"{kind.upper()}_h{Hz}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=multihorizon_pinball_loss(quantiles))
    return model


# ============================================================== gradient boosting

def xgb_features(inputs: List[np.ndarray], horizon: int) -> np.ndarray:
    """Flat design matrix for one horizon: all lag history + that horizon's covariates.

    `inputs` is what `build_dataset` returns: 17 history arrays then da_future then
    calendar. XGBoost has no notion of a horizon axis, so a separate model is fitted
    per horizon and each one sees only its own known-future slice.
    """
    hist = [x.reshape(len(x), -1) for x in inputs[:17]]
    da, cal = inputs[-2], inputs[-1]
    return np.concatenate(hist + [da[:, horizon - 1, :], cal[:, horizon - 1, :]],
                          axis=1).astype("float32")


def fit_xgb_horizon(Xtr, ytr, Xva, yva, quantiles: Sequence[float],
                    n_estimators: int = 200, max_depth: int = 4,
                    learning_rate: float = 0.1, seed: int = 42):
    """One multi-quantile booster for one horizon. Returns (N, Q) predictions helper.

    XGBoost's `reg:quantileerror` takes a vector of alphas and emits one column per
    alpha, so a single booster covers all five quantiles. It does NOT enforce
    monotonicity across them, so this model can and does produce crossing -- which is
    the honest behaviour of a baseline without a hierarchical head.
    """
    import xgboost as xgb

    m = xgb.XGBRegressor(objective="reg:quantileerror",
                         quantile_alpha=np.asarray(quantiles, dtype=float),
                         n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, subsample=0.8,
                         colsample_bytree=0.8, random_state=seed,
                         early_stopping_rounds=20, verbosity=0, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m
