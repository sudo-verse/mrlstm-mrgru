"""MR-LSTM / MR-GRU: market-rule-informed recurrent networks.

Identical to MRINN except for the encoder stage. MRINN encodes each market
signal with a Dense stack over a flat (len(lags),) vector; here each signal
gets its own LSTM or GRU over a (T, 1) sequence. Everything downstream --
the balancing-energy, market-reference and scarcity price rules, the smooth
min/max combination and the hierarchical quantile head -- is imported
unchanged from library_imbalance.model.
"""

import time
from typing import Sequence

import tensorflow as tf
from tensorflow.keras import layers, models

from . import _mrinn_path  # noqa: F401  (side effect: sys.path)
from .seqdata import FEATURE_SPEC, assert_input_alignment, to_sequence_inputs

# The market rules, imported rather than reimplemented.
from library_imbalance.model import (  # noqa: E402
    HierarchicalQuantileHeadQ50,
    gate_first,
    gate_second,
    get_P_EX,
    get_P_RE,
    get_P_SC,
    multi_quantile_pinball_loss,
    safe_div_pair,
    smooth_max,
    smooth_min,
)

CUSTOM_OBJECTS = {
    "safe_div_pair": safe_div_pair,
    "gate_first": gate_first,
    "gate_second": gate_second,
}

_CELLS = {"lstm": layers.LSTM, "gru": layers.GRU}


def _recurrent_encoder(x, hidden_units: int, num_layer: int, cell: str, name: str):
    """(N, T, 1) -> (N, hidden_units), the shape the rule blocks expect."""
    Cell = _CELLS[cell]
    for i in range(num_layer):
        is_last = i == num_layer - 1
        x = Cell(
            hidden_units,
            return_sequences=not is_last,
            name=f"{name}_{cell}_{i}",
        )(x)
    return x


def build_MRRNN(CASE_FOR_OUTPUT, X_train, y_train, X_val, y_val,
                hidden_units, num_layer, epochs,
                C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10,
                cell="lstm",
                lags: Sequence[int] = (1,),
                quantiles=(0.1, 0.5, 0.9),
                checkpoint_path="best_mrrnn.keras",
                batch_size=1024,
                learning_rate=1e-3,
                verbose=0):
    """Build and fit an MR-LSTM (cell='lstm') or MR-GRU (cell='gru').

    Mirrors library_imbalance.model.build_MRINN, including the four
    CASE_FOR_OUTPUT ablation branches, and returns (model, history).
    """
    cell = cell.lower()
    if cell not in _CELLS:
        raise ValueError(f"cell must be one of {sorted(_CELLS)}, got {cell!r}")
    if CASE_FOR_OUTPUT not in {
        "ReservePriceIndex", "ExchangePriceIndex", "ScarcityFunction", "ImbalancePrice"
    }:
        raise ValueError(f"unknown CASE_FOR_OUTPUT: {CASE_FOR_OUTPUT!r}")

    lags = sorted(int(L) for L in lags)
    T = len(lags)
    quantiles = list(quantiles)

    # ---- inputs: (T, 1) per signal, names matched to MRINN ----
    feats_in = [
        layers.Input(shape=(T, 1), name=input_name)
        for _, input_name in FEATURE_SPEC
    ]

    # ---- encoder: one recurrent stack per signal (replaces MRINN's Dense stacks) ----
    #
    # An encoder is built for all 17 signals, but Keras prunes L_DA's: get_P_EX
    # never consumes L_DA_rep, deriving w_DA = 1 - w_ID15 - w_ID60 instead, and
    # data.load_data pins the L_DA column to 0. So 16 encoders carry parameters.
    # The L_DA input stays on the model and must still be fed. Inherited from
    # MRINN -- kept as-is so the comparison stays a clean ablation.
    reps = [
        _recurrent_encoder(x, hidden_units, num_layer, cell, name=base)
        for x, (base, _) in zip(feats_in, FEATURE_SPEC)
    ]

    (system_imbalance_rep,
     E_aFRR_pos_rep, E_mFRR_pos_rep, P_aFRR_pos_rep, P_mFRR_pos_rep,
     E_aFRR_neg_rep, E_mFRR_neg_rep, P_aFRR_neg_rep, P_mFRR_neg_rep,
     P_VoAA_pos_rep, P_VoAA_neg_rep,
     P_ID15_nemo_rep, P_ID60_nemo_rep, P_DA_nemo_rep,
     L_ID15_rep, L_ID60_rep, L_DA_rep) = reps

    # ---- market rules: unchanged from MRINN ----
    if CASE_FOR_OUTPUT == "ReservePriceIndex":
        head_in = get_P_RE(
            E_aFRR_pos_rep, E_mFRR_pos_rep, E_aFRR_neg_rep, E_mFRR_neg_rep,
            system_imbalance_rep,
            P_aFRR_pos_rep, P_mFRR_pos_rep, P_aFRR_neg_rep, P_mFRR_neg_rep,
            P_VoAA_pos_rep, P_VoAA_neg_rep, hidden_units)

    elif CASE_FOR_OUTPUT == "ExchangePriceIndex":
        head_in, _ = get_P_EX(
            P_ID15_nemo_rep, L_ID15_rep, P_ID60_nemo_rep, L_ID60_rep, P_DA_nemo_rep,
            system_imbalance_rep, C0, C1, C2, C3, C4, C5, C6)

    elif CASE_FOR_OUTPUT == "ScarcityFunction":
        _, P_base = get_P_EX(
            P_ID15_nemo_rep, L_ID15_rep, P_ID60_nemo_rep, L_ID60_rep, P_DA_nemo_rep,
            system_imbalance_rep, C0, C1, C2, C3, C4, C5, C6)
        head_in = get_P_SC(P_base, system_imbalance_rep, C7, C8, C9, C10)

    else:  # ImbalancePrice
        P_RE = get_P_RE(
            E_aFRR_pos_rep, E_mFRR_pos_rep, E_aFRR_neg_rep, E_mFRR_neg_rep,
            system_imbalance_rep,
            P_aFRR_pos_rep, P_mFRR_pos_rep, P_aFRR_neg_rep, P_mFRR_neg_rep,
            P_VoAA_pos_rep, P_VoAA_neg_rep, hidden_units)

        P_EX, P_base = get_P_EX(
            P_ID15_nemo_rep, L_ID15_rep, P_ID60_nemo_rep, L_ID60_rep, P_DA_nemo_rep,
            system_imbalance_rep, C0, C1, C2, C3, C4, C5, C6)

        P_SC = get_P_SC(P_base, system_imbalance_rep, C7, C8, C9, C10)

        term_final_min = smooth_min(smooth_min(P_RE, P_EX), P_SC)
        term_final_max = smooth_max(smooth_max(P_RE, P_EX), P_SC)

        W_final = layers.Dense(2, activation="softmax", name="final_gate")(system_imbalance_rep)
        head_in = layers.Add(name="imbalance_price_rep")([
            layers.Multiply()([term_final_min, W_final[:, 0:1]]),
            layers.Multiply()([term_final_max, W_final[:, 1:2]]),
        ])

    output = HierarchicalQuantileHeadQ50(
        head_in, quantiles, name_prefix="imbalance_price_hq")

    model_name = f"MR{cell.upper()}_{CASE_FOR_OUTPUT}"
    model = models.Model(inputs=feats_in, outputs=output, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=multi_quantile_pinball_loss(quantiles),
    )

    print(f"{model_name} | cell={cell} T={T} hidden={hidden_units} layers={num_layer}")
    print("Total parameters:", model.count_params())

    train_inputs = to_sequence_inputs(X_train, lags=lags)
    val_inputs = to_sequence_inputs(X_val, lags=lags)
    assert_input_alignment(model, train_inputs)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    train_start = time.perf_counter()
    history = model.fit(
        train_inputs,
        y_train.to_numpy(dtype="float32"),
        validation_data=(val_inputs, y_val.to_numpy(dtype="float32")),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt],
        verbose=verbose,
    )
    print(f"Training time: {time.perf_counter() - train_start:.2f}s "
          f"({(time.perf_counter() - train_start) / max(epochs, 1):.2f}s/epoch)")

    return model, history


def make_inference(X_test, lags, quantiles, checkpoint_path="best_mrrnn.keras"):
    """Load the best checkpoint and predict; returns a list of Q arrays."""
    lags = sorted(int(L) for L in lags)

    best_model = tf.keras.models.load_model(
        checkpoint_path, custom_objects=CUSTOM_OBJECTS, compile=False)

    test_inputs = to_sequence_inputs(X_test, lags=lags)
    assert_input_alignment(best_model, test_inputs)

    start = time.perf_counter()
    y_pred_scaled = best_model.predict(test_inputs, verbose=0, batch_size=1024)  # (N, Q)
    print("Inference time:", time.perf_counter() - start)

    return [y_pred_scaled[:, j] for j in range(len(quantiles))]
