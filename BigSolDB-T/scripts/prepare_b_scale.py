"""
Per-split Van't Hoff slope scale, fitted on TRAINING ROWS ONLY.

`b_scale` is the constant folded into the temperature variable of model Bc
(see docs/vanthoff_conventions.md). It must never see validation or test data,
so every fit here is restricted to the row indices of a split's training set.

Reconstructed from the API used by notebooks 06 and 08. Eligibility
(>= 3 rows AND >= 3 distinct temperatures per pair) reproduces the pair counts
reported in notebook 07: 10,500 pairs over the full dataset, and 10,488 under
the >= 4 distinct-temperature sensitivity check.

Run:  python3 scripts/prepare_b_scale.py      ->  results/b_scale.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

T_REF = 298.15
MIN_ROWS_PER_PAIR = 3
MIN_DISTINCT_T = 3
SPLIT_NAMES = ("random", "textrap", "coldsol", "coldpair", "coldsolv")


def per_pair_slopes(T, y, pair_id, idx=None,
                    min_rows=MIN_ROWS_PER_PAIR, min_distinct_T=MIN_DISTINCT_T):
    """Per-pair OLS Van't Hoff fit in MODEL convention.

        y = a + b * (1/T_ref - 1/T)

    so b > 0 means solubility rises with temperature. Only rows in `idx` are
    used (pass the split's training indices to keep the fit leak-free).

    Returns (slopes, pair_ids, r2) as parallel arrays, one entry per eligible
    pair. `len(slopes)` is the number of pairs that met the eligibility rule.
    """
    T = np.asarray(T, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pair_id = np.asarray(pair_id)

    if idx is not None:
        idx = np.asarray(idx)
        T, y, pair_id = T[idx], y[idx], pair_id[idx]

    z = 1.0 / T_REF - 1.0 / T                      # model-convention abscissa
    order = np.argsort(pair_id, kind="stable")
    pid_s, z_s, y_s = pair_id[order], z[order], y[order]
    bounds = np.concatenate([[0], np.where(np.diff(pid_s) != 0)[0] + 1, [len(pid_s)]])

    slopes, pids, r2s = [], [], []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < min_rows:
            continue
        zz, yy = z_s[lo:hi], y_s[lo:hi]
        if np.unique(zz).size < min_distinct_T:
            continue
        zc, yc = zz - zz.mean(), yy - yy.mean()
        den = float((zc ** 2).sum())
        if den <= 0:
            continue
        b = float((zc * yc).sum() / den)
        resid = yc - b * zc
        ss_tot = float((yc ** 2).sum())
        slopes.append(b)
        pids.append(int(pid_s[lo]))
        r2s.append(1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan)

    return np.array(slopes), np.array(pids, dtype=np.int64), np.array(r2s)


def b_scale_from_train(T, y, pair_id, train_idx):
    """Robust slope scale: median |b| over eligible TRAINING pairs."""
    slopes, _, _ = per_pair_slopes(T, y, pair_id, train_idx)
    return float(np.median(np.abs(slopes))), int(len(slopes))


def main():
    feats = np.load(RES / "features.npz", allow_pickle=True)
    T, y, pair_id = feats["T"], feats["y"], feats["pair_id"]
    splits = np.load(RES / "splits.npz")

    out = {
        "meta": {
            "T_ref_K": T_REF,
            "convention": "y = a + b * (1/T_ref - 1/T);  b > 0 => solubility rises with T",
            "definition": "b_scale = median |b| of per-pair OLS fits over TRAINING ROWS ONLY",
            "eligibility": {
                "min_rows_per_pair": MIN_ROWS_PER_PAIR,
                "min_distinct_temperatures": MIN_DISTINCT_T,
            },
        },
        "splits": {},
    }

    for split in SPLIT_NAMES:
        tr = splits[f"{split}_train"]
        b_scale, n_used = b_scale_from_train(T, y, pair_id, tr)
        out["splits"][split] = {
            "b_scale": b_scale,
            "n_pairs_used": n_used,
            "n_pairs_in_train": int(np.unique(pair_id[tr]).size),
            "n_train_rows": int(len(tr)),
        }
        print(f"{split:<9} b_scale = {b_scale:8.2f} K   "
              f"from {n_used:,} eligible pairs of {out['splits'][split]['n_pairs_in_train']:,} in train")

    path = RES / "b_scale.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
