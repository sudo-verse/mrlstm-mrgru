"""Put the MRINN checkout on sys.path.

MRINN is treated as a dependency, not as vendored source: this repo imports
its data pipeline, market-rule blocks and evaluation metrics rather than
copying them. The checkout is left untouched so it stays diffable against
upstream (runyao-yu/MRINN), which is what keeps MR-LSTM / MR-GRU a clean
encoder-only ablation.

Get it with::

    git clone https://github.com/runyao-yu/MRINN external/MRINN

Resolution is relative to this file, not the working directory, so the
package works from a notebook opened anywhere. Candidates are tried in
order; set the MRINN_ROOT environment variable to override.
"""

import os
import sys
from pathlib import Path

# .../MRLSTM/library_mrrnn/_mrinn_path.py -> parents[1] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]

# In order of preference. The last entry is the sibling layout used during
# development (fyp/MRLSTM next to fyp/mrinn/MRINN).
_CANDIDATES = [
    _REPO_ROOT / "external" / "MRINN",
    _REPO_ROOT / "MRINN",
    _REPO_ROOT.parent / "MRINN",
    _REPO_ROOT.parent / "mrinn" / "MRINN",
]


def _is_checkout(path: Path) -> bool:
    return (path / "library_imbalance").is_dir()


def _resolve_mrinn_root() -> Path:
    override = os.environ.get("MRINN_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    for candidate in _CANDIDATES:
        if _is_checkout(candidate):
            return candidate.resolve()

    # Nothing found: return the canonical location so the error message in
    # ensure_on_path() names the place the setup instructions tell you to use.
    return _CANDIDATES[0]


MRINN_ROOT = _resolve_mrinn_root()


def ensure_on_path() -> Path:
    if not _is_checkout(MRINN_ROOT):
        searched = "\n  ".join(str(c) for c in _CANDIDATES)
        raise ImportError(
            f"MRINN checkout not found at {MRINN_ROOT}.\n"
            "Clone it with:\n"
            "  git clone https://github.com/runyao-yu/MRINN external/MRINN\n"
            "or set the MRINN_ROOT environment variable to the directory "
            "containing library_imbalance/.\n"
            f"Searched:\n  {searched}"
        )

    root = str(MRINN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return MRINN_ROOT


ensure_on_path()
