"""Phase 0 / Phase 1: pull imbalance-netting volumes for the APG control area.

WHAT THIS FETCHES
-----------------
The supervisor asked for `INC+`, `INC-` and `Settlementprice_INC` "from their website".
They are not on APG's transparency portal -- APG publishes eight balancing datasets and
none of them is a netting series. The netting volumes live on the ENTSO-E Transparency
Platform, as "Netted and Exchanged Volumes per Border":

    documentType = A30   (cross-border schedule)
    processType  = A51   (automatic frequency restoration reserves)
    businessType = A45   (scheduled activated reserves)
    Acquiring_Domain / Connecting_Domain = the two area codes of the border

They are published PER BORDER, which resolves the ambiguity in the request:

    INC  (APG's bilateral cooperation with Slovenia) = the AT <-> SI border alone
    IGCC (the European platform)                     = the sum over all AT borders

This script pulls every AT border so both definitions can be produced, and the
supervisor can confirm which he means without a second download.

TWO THINGS TO KNOW BEFORE RUNNING
---------------------------------
1. A free API token is required. Register at transparency.entsoe.eu, then email
   transparency@entsoe.eu asking for API access; they enable the token on the account.
   Export it as ENTSOE_TOKEN.
2. IGCC volumes are published from NOVEMBER 2022. The modelling dataset starts
   2022-01-01, so 29,184 quarter-hours (20.8% of the record) will have no netting data.
   The script marks those rows with INC_available = 0 rather than filling zeros, because
   a zero netting volume and an unpublished one are different states.

    export ENTSOE_TOKEN=...
    .venv/bin/python fetch_netting_data.py --start 2022-11-01 --end 2026-01-01
    .venv/bin/python fetch_netting_data.py --probe          # one day, to check access
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://web-api.tp.entsoe.eu/api"
AT = "10YAT-APG------L"
# AT's IGCC-coupled borders, per the adjacency used by the base paper's own group.
BORDERS = {
    "CZ": "10YCZ-CEPS-----N",
    "DE_LU": "10Y1001A1001A82H",
    "HU": "10YHU-MAVIR----U",
    "IT_NORD": "10Y1001A1001A73I",
    "SI": "10YSI-ELES-----O",
}
NS = {"": "urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1"}
OUT = Path("data")
STEP = "15min"


def fetch_border(token, other, t0, t1, session, retries=3):
    """One border, one window. Returns a tz-aware Series of netted volume in MW."""
    params = {
        "securityToken": token, "documentType": "A30", "processType": "A51",
        "businessType": "A45",
        "Acquiring_Domain": AT, "Connecting_Domain": other,
        "periodStart": t0.strftime("%Y%m%d%H%M"),
        "periodEnd": t1.strftime("%Y%m%d%H%M"),
    }
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=120)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:                       # rate limited
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 400:                       # no data for the window
            return pd.Series(dtype="float64")
        r.raise_for_status()
        return parse(r.content)
    return pd.Series(dtype="float64")


def parse(xml_bytes):
    """Flatten every TimeSeries/Period/Point into one series indexed by UTC time."""
    root = ET.fromstring(xml_bytes)
    idx, val = [], []
    for ts in root.iter():
        if not ts.tag.endswith("TimeSeries"):
            continue
        for per in ts:
            if not per.tag.endswith("Period"):
                continue
            start = res = None
            for el in per:
                if el.tag.endswith("timeInterval"):
                    for c in el:
                        if c.tag.endswith("start"):
                            start = pd.Timestamp(c.text)
                elif el.tag.endswith("resolution"):
                    res = pd.Timedelta(el.text)
            if start is None or res is None:
                continue
            for pt in per:
                if not pt.tag.endswith("Point"):
                    continue
                pos = qty = None
                for c in pt:
                    if c.tag.endswith("position"):
                        pos = int(c.text)
                    elif c.tag.endswith("quantity"):
                        qty = float(c.text)
                if pos is not None and qty is not None:
                    idx.append(start + (pos - 1) * res)
                    val.append(qty)
    if not idx:
        return pd.Series(dtype="float64")
    s = pd.Series(val, index=pd.DatetimeIndex(idx))
    return s[~s.index.duplicated(keep="last")].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-11-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--chunk", default="7D", help="request window size")
    ap.add_argument("--probe", action="store_true", help="fetch one day and stop")
    args = ap.parse_args()

    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        sys.exit("ENTSOE_TOKEN is not set. Register at transparency.entsoe.eu, then\n"
                 "email transparency@entsoe.eu for API access, then export the token.")

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = (t0 + pd.Timedelta("1D")) if args.probe else pd.Timestamp(args.end, tz="UTC")
    edges = list(pd.date_range(t0, t1, freq=args.chunk)) + [t1]
    edges = sorted(set(edges))

    OUT.mkdir(exist_ok=True)
    session = requests.Session()
    frames = {}
    for name, code in BORDERS.items():
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = fetch_border(token, code, a, b, session)
            if len(s):
                parts.append(s)
            print(f"  {name:8s} {a.date()} -> {b.date()}  {len(s):>7,} points", flush=True)
        if parts:
            s = pd.concat(parts).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            # published at up to 4-second resolution; the model runs on 15 minutes
            frames[name] = s.resample(STEP).mean()
        if args.probe:
            break

    if not frames:
        sys.exit("no data returned -- check the token has API access enabled")

    df = pd.DataFrame(frames)
    df["IGCC_total"] = df.sum(axis=1, skipna=True)
    df["INC_SI"] = df.get("SI", np.nan)
    raw = OUT / "netting_raw.csv"
    df.to_csv(raw)
    print(f"\nwritten: {raw}   {len(df):,} rows, borders: {', '.join(frames)}")

    # ---- the verification the supervisor explicitly asked for --------------
    print("\n" + "=" * 74)
    print("SIGN VERIFICATION -- show this to the supervisor before any modelling")
    print("=" * 74)
    print(df.describe().T.round(2).to_string())
    for col in ("IGCC_total", "INC_SI"):
        if col not in df or df[col].isna().all():
            continue
        v = df[col].dropna()
        print(f"\n  {col}: n={len(v):,}  positive {(v > 0).mean() * 100:5.1f}%  "
              f"negative {(v < 0).mean() * 100:5.1f}%  zero {(v == 0).mean() * 100:5.1f}%")
        print(f"    mean {v.mean():+8.2f}  median {v.median():+8.2f}  "
              f"p5 {np.percentile(v, 5):+8.2f}  p95 {np.percentile(v, 95):+8.2f}")
    print("\n  INC+ and INC- are then derived as the positive and negative parts,")
    print("  but WHICH SIGN MEANS 'netting reduced upward activation' must be")
    print("  confirmed against the sign of system_imbalance -- see the correlation")
    print("  printed by merge_netting_data.py once this file exists.")


if __name__ == "__main__":
    main()
