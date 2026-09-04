"""Phase 0.4-0.6 / Phase 1: merge netting volumes and answer the open questions.

Run after fetch_netting_data.py has produced data/netting_raw.csv. This script does
the three checks that decide whether the netting block is worth building, then writes
the extended dataset.

  0.3  DOUBLE COUNTING. Netting happens BEFORE activation, so if the published
       activation volumes are already net of it, subtracting INC again removes the same
       effect twice. Tested by regressing net activation on the system imbalance with
       and without the netting term: if netting is already inside the activation data,
       INC will add nothing and may carry the wrong sign.
  0.4  REDUNDANCY. `system_imbalance` is already 83.1% explained (r = +0.911) by net
       activation. If INC regresses on the existing 17 features at R^2 > 0.9 it is a
       renamed copy of something the model already sees.
  0.6  COVERAGE. Rows before the IGCC publication start get INC_available = 0 rather
       than a zero volume, because "no netting" and "not published" are different.

    .venv/bin/python merge_netting_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMB = Path("external/MRINN/Data/imbalance_data.csv")
RAW = Path("data/netting_raw.csv")
OUT = Path("data/imbalance_data_with_inc.csv")
FEATS = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg", "P_aFRR_pos_MOL",
         "P_aFRR_neg_MOL", "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo", "L_ID15",
         "L_ID60", "system_imbalance", "E_aFRR_pos", "E_mFRR_pos", "E_aFRR_neg",
         "E_mFRR_neg"]


def main():
    if not RAW.exists():
        sys.exit(f"{RAW} not found -- run fetch_netting_data.py first "
                 "(needs ENTSOE_TOKEN).")
    d = pd.read_csv(IMB)
    d["t"] = pd.to_datetime(d["Time [UTC] start"], utc=True)
    n = pd.read_csv(RAW, index_col=0, parse_dates=True)
    if n.index.tz is None:
        n.index = n.index.tz_localize("UTC")

    which = "INC_SI" if "INC_SI" in n.columns else "IGCC_total"
    m = d.merge(n[[which]].rename(columns={which: "INC"}),
                left_on="t", right_index=True, how="left")
    m["INC_available"] = m.INC.notna().astype(int)
    # INC+ / INC- as the two signed parts; the sign convention is confirmed below.
    m["INC_plus"] = m.INC.clip(lower=0).fillna(0.0)
    m["INC_minus"] = (-m.INC.clip(upper=0)).fillna(0.0)

    print(f"merged on {which}: {m.INC_available.sum():,} of {len(m):,} rows have netting "
          f"({m.INC_available.mean() * 100:.1f}%)\n")

    ok = m.INC_available == 1
    V = m.system_imbalance.values
    net = ((m.E_aFRR_pos + m.E_mFRR_pos) - (m.E_aFRR_neg + m.E_mFRR_neg)).values

    print("=" * 74)
    print("0.3  DOUBLE COUNTING -- is the published activation already net of IGCC?")
    print("=" * 74)
    a = np.polyfit(V[ok], net[ok], 1)
    print(f"  net activation ~ V              slope {a[0]:.4f}   r {np.corrcoef(net[ok], V[ok])[0, 1]:+.4f}")
    X = np.c_[V[ok], m.INC.values[ok]]
    b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X], net[ok], rcond=None)[0]
    print(f"  net activation ~ V + INC        V {b[1]:+.4f}   INC {b[2]:+.4f}")
    print(f"  corr(INC, V) = {np.corrcoef(m.INC.values[ok], V[ok])[0, 1]:+.4f}")
    print("\n  READ THIS AS: if the INC coefficient is near zero, the activation volumes")
    print("  already absorb netting and the proposed subtraction would double-count.")
    print("  If it is materially negative, netting is genuinely additional information.")

    print("\n" + "=" * 74)
    print("0.4  REDUNDANCY -- is INC already visible through the existing 17 features?")
    print("=" * 74)
    A = m.loc[ok, FEATS].fillna(0.0).to_numpy()
    A = np.c_[np.ones(len(A)), A]
    y = m.INC.values[ok]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - np.var(y - A @ coef) / max(np.var(y), 1e-12)
    print(f"  R^2 of INC on the existing features: {r2:.4f}")
    verdict = ("REDUNDANT -- stop and report" if r2 > 0.9 else
               "partially redundant -- proceed but expect a small effect" if r2 > 0.6 else
               "genuinely new information -- proceed")
    print(f"  verdict: {verdict}")

    print("\n" + "=" * 74)
    print("SIGN CONVENTION -- for the supervisor to confirm")
    print("=" * 74)
    s, l = ok & (m.system_imbalance > 0), ok & (m.system_imbalance < 0)
    print(f"  mean INC when system SHORT (V>0): {m.INC[s].mean():+8.2f}")
    print(f"  mean INC when system LONG  (V<0): {m.INC[l].mean():+8.2f}")
    print("  A netting volume that offsets the imbalance should carry the OPPOSITE sign")
    print("  to V. If it carries the same sign, INC_plus/INC_minus must be swapped.")

    m.drop(columns=["t"]).to_csv(OUT, index=False)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
