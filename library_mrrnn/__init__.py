"""MR-LSTM / MR-GRU: market-rule-informed recurrent networks.

Extends MRINN (Yu et al., Adv. Eng. Informatics 76 (2026) 105083) by
replacing its per-feature Dense encoders with per-feature LSTM/GRU
encoders over a real time axis. The market-rule head is imported from
the MRINN checkout unchanged, so the comparison is a clean ablation.

Data loading, splitting, lag construction, constant scaling and all
evaluation metrics are re-exported from library_imbalance.
"""

from . import _mrinn_path  # noqa: F401  (side effect: sys.path)
from ._mrinn_path import MRINN_ROOT

# --- reused unchanged from the MRINN checkout ---
from library_imbalance.data import (  # noqa: E402
    load_data,
    scale_data,
    scaled_params,
    shift_data,
    split_data,
)
from library_imbalance.evaluation import (  # noqa: E402
    create_gif_from_figures,
    evaluate_performance,
    naive_baseline,
    naive_reference_baseline,
    plot_prediction_vs_true,
    plot_train_val_loss,
    prepare_prediction_arrays,
    save_inference_figures,
)
from library_imbalance.model import set_random_seed  # noqa: E402

# --- new here ---
from .model import build_MRRNN, make_inference  # noqa: E402
from .seqdata import (  # noqa: E402
    FEATURE_BASES,
    FEATURE_SPEC,
    INPUT_NAMES,
    assert_input_alignment,
    scale_data_shared_lags,
    to_sequence_inputs,
)

__all__ = [
    # data
    "load_data", "split_data", "shift_data", "scale_data", "scaled_params",
    "scale_data_shared_lags", "to_sequence_inputs",
    # model
    "set_random_seed", "build_MRRNN", "make_inference",
    # introspection / guards
    "FEATURE_SPEC", "FEATURE_BASES", "INPUT_NAMES", "assert_input_alignment",
    "MRINN_ROOT",
    # evaluation
    "evaluate_performance", "naive_baseline", "naive_reference_baseline",
    "prepare_prediction_arrays", "plot_prediction_vs_true", "plot_train_val_loss",
    "save_inference_figures", "create_gif_from_figures",
]
__version__ = "0.1.0"
