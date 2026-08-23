"""Sequence-shaped inputs for the market-rule recurrent models.

MRINN feeds each market signal to a Dense stack as a flat ``(len(lags),)``
vector. A recurrent encoder needs a real time axis, oldest step first, so
this module reshapes the same lag columns into ``(N, T, 1)`` per feature.
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class AsinhScaler:
    def __init__(self):
        self.scale_ = None
        
    def fit(self, X, y=None):
        import numpy as np
        X = np.asarray(X)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        self.scale_ = q75 - q25
        if isinstance(self.scale_, np.ndarray):
            self.scale_[self.scale_ == 0] = 1.0
        elif self.scale_ == 0:
            self.scale_ = 1.0
        return self

    def transform(self, X, y=None):
        import numpy as np
        return np.arcsinh(np.asarray(X) / self.scale_)

    def inverse_transform(self, X, y=None):
        import numpy as np
        return np.sinh(np.asarray(X)) * self.scale_

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg",
                "P_aFRR_pos_MOL", "P_aFRR_neg_MOL",
                "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo", "imbalance_price"]


from . import _mrinn_path  # noqa: F401  (side effect: sys.path)


# (dataframe column base, model input name) in the exact order of feats_in
# in library_imbalance.model.build_MRINN. The order is positional -- the
# rule blocks read these tensors by position, so a permutation silently
# trains a wrong model.
#
# P_aFRR_pos_MOL / P_aFRR_neg_MOL fill the P_VoAA_* slots: data.load_data
# renames P_VoAA_pos/neg to those names on read.
FEATURE_SPEC: List[Tuple[str, str]] = [
    ("system_imbalance", "system_imbalance"),
    ("E_aFRR_pos", "E_aFRR_pos_in"),
    ("E_mFRR_pos", "E_mFRR_pos_in"),
    ("P_aFRR_pos", "P_aFRR_pos_in"),
    ("P_mFRR_pos", "P_mFRR_pos_in"),
    ("E_aFRR_neg", "E_aFRR_neg_in"),
    ("E_mFRR_neg", "E_mFRR_neg_in"),
    ("P_aFRR_neg", "P_aFRR_neg_in"),
    ("P_mFRR_neg", "P_mFRR_neg_in"),
    ("P_aFRR_pos_MOL", "P_VoAA_pos_in"),
    ("P_aFRR_neg_MOL", "P_VoAA_neg_in"),
    ("P_ID15_nemo", "P_ID15_nemo_in"),
    ("P_ID60_nemo", "P_ID60_nemo_in"),
    ("P_DA_nemo", "P_DA_nemo_in"),
    ("L_ID15", "L_ID15_in"),
    ("L_ID60", "L_ID60_in"),
    ("L_DA", "L_DA_in"),
]

FEATURE_BASES: List[str] = [base for base, _ in FEATURE_SPEC]
INPUT_NAMES: List[str] = [name for _, name in FEATURE_SPEC]


def _stack_lags_seq(frame: pd.DataFrame, base: str, lags: Sequence[int]) -> np.ndarray:
    """One feature as (N, T, 1), oldest timestep first.

    Column ``{base}_lag1`` is the most recent step (data.make_shifts uses
    ``shift(L)``), so the lag axis is reversed before reshaping.
    """
    cols = [f"{base}_lag{L}" for L in lags]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing lag columns for '{base}': {missing}")

    arr = frame[cols].to_numpy(dtype="float32")        # (N, T) newest -> oldest
    arr = np.ascontiguousarray(arr[:, ::-1])           # oldest -> newest
    return arr[:, :, np.newaxis]                       # (N, T, 1)


def to_sequence_inputs(frame: pd.DataFrame, lags: Sequence[int]) -> List[np.ndarray]:
    """The 17 model inputs, each (N, T, 1), in FEATURE_SPEC order."""
    lags = sorted(int(L) for L in lags)
    return [_stack_lags_seq(frame, base, lags) for base in FEATURE_BASES]


def _split_lag_column(col: str) -> Tuple[str, bool]:
    """('P_DA_nemo_lag3') -> ('P_DA_nemo', True); non-lag columns -> (col, False)."""
    if "_lag" not in col:
        return col, False
    base, _, suffix = col.rpartition("_lag")
    if not suffix.isdigit():
        return col, False
    return base, True


def scale_data_shared_lags(shifted_train, shifted_val, shifted_test,
                           feature_names: List[str], target_col):
    """Like data.scale_data, but one scaler per signal instead of per column.

    data.scale_data fits a RobustScaler per column, so P_DA_nemo_lag1 and
    P_DA_nemo_lag32 get different medians and IQRs -- which warps the time
    axis a recurrent encoder is trying to read. Here every lag column of a
    signal shares one scaler, fit on all of that signal's lags stacked
    together (the idiom in data.scale_param).

    Returns the same tuple as data.scale_data.
    """
    groups = {}
    for col in feature_names:
        base, is_lag = _split_lag_column(col)
        key = base if is_lag else col
        groups.setdefault(key, []).append(col)

    Xtr = shifted_train[feature_names].to_numpy(dtype=float).copy()
    Xva = shifted_val[feature_names].to_numpy(dtype=float).copy()
    Xte = shifted_test[feature_names].to_numpy(dtype=float).copy()

    index_of = {c: i for i, c in enumerate(feature_names)}

    for base, cols in groups.items():
        idx = [index_of[c] for c in cols]
        stacked = Xtr[:, idx].reshape(-1, 1)
        scaler = RobustScaler().fit(stacked)
        for i in idx:
            Xtr[:, i] = scaler.transform(Xtr[:, i].reshape(-1, 1)).ravel()
            Xva[:, i] = scaler.transform(Xva[:, i].reshape(-1, 1)).ravel()
            Xte[:, i] = scaler.transform(Xte[:, i].reshape(-1, 1)).ravel()

    ytr = shifted_train[target_col].to_numpy(dtype=float).reshape(-1, 1)
    yva = shifted_val[target_col].to_numpy(dtype=float).reshape(-1, 1)
    yte = shifted_test[target_col].to_numpy(dtype=float).reshape(-1, 1)

    y_scaler = RobustScaler().fit(ytr)

    X_train = pd.DataFrame(Xtr, columns=feature_names)
    X_val = pd.DataFrame(Xva, columns=feature_names)
    X_test = pd.DataFrame(Xte, columns=feature_names)
    y_train = pd.DataFrame(y_scaler.transform(ytr).ravel(), columns=target_col)
    y_val = pd.DataFrame(y_scaler.transform(yva).ravel(), columns=target_col)
    y_test = pd.DataFrame(y_scaler.transform(yte).ravel(), columns=target_col)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler


def assert_input_alignment(model, inputs: List[np.ndarray]) -> None:
    """Fail loudly if the input arrays do not line up with the model's inputs.

    Positional misalignment is the highest-impact silent bug here: the model
    still trains, it just feeds liquidity into a price slot.
    """
    model_names = [t.name.split(":")[0] for t in model.inputs]

    if len(inputs) != len(model_names):
        raise ValueError(
            f"got {len(inputs)} input arrays for {len(model_names)} model inputs"
        )

    if model_names != INPUT_NAMES:
        mismatch = [
            (i, a, b) for i, (a, b) in enumerate(zip(model_names, INPUT_NAMES)) if a != b
        ]
        raise ValueError(f"input name order mismatch at {mismatch}")

    for name, arr in zip(model_names, inputs):
        if arr.ndim != 3 or arr.shape[-1] != 1:
            raise ValueError(f"input '{name}' has shape {arr.shape}, expected (N, T, 1)")
