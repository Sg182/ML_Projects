"""
Persist the joint "unseen solute + T-extrapolation" split used by notebook 08.

Notebook 08 built this split inline, so nothing on disk described it. Notebook 10
needs the identical partition to compare GINE against the MLP arm, so the same
construction is reproduced verbatim here and cached.

Construction (from 08_distribution_shift_stress_tests, JOINT_SPLIT_SEED = 42):
  1. group_split over solutes (80/10/10, seed 42) -> train / val / test solutes
  2. for each test-solute pair with >= 3 rows and dT >= 20 K, keep only rows
     above T_min + 0.75 * (T_max - T_min), i.e. the upper 25 % of its T range
  3. train = all rows of train solutes; val = all rows of val solutes

Run:  python3 scripts/make_joint_split.py   ->  results/splits_joint.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
JOINT_SPLIT_SEED = 42


def group_split(group_ids, n_groups, seed, frac_train=0.80, frac_val=0.10):
    """Identical to the helper in notebooks 03 and 08."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)
    n_tr = int(round(frac_train * n_groups))
    n_va = int(round(frac_val * n_groups))
    tr_g, va_g = set(order[:n_tr].tolist()), set(order[n_tr:n_tr + n_va].tolist())
    te_g = set(order[n_tr + n_va:].tolist())
    tr = np.array([g in tr_g for g in group_ids])
    va = np.array([g in va_g for g in group_ids])
    te = np.array([g in te_g for g in group_ids])
    return np.where(tr)[0], np.where(va)[0], np.where(te)[0], (len(tr_g), len(va_g), len(te_g))


def main():
    feats = np.load(RES / "features.npz", allow_pickle=True)
    T_arr, pair_id = feats["T"].astype(np.float64), feats["pair_id"]
    df = pd.read_csv(ROOT / "data" / "BigSolDBv2.0.csv").dropna(
        subset=["LogS(mol/L)"]).reset_index(drop=True)
    solute_id, uniq = pd.factorize(df["SMILES_Solute"].astype(str))

    tr_idx, va_idx, te_solute_rows, gc = group_split(solute_id, len(uniq), seed=JOINT_SPLIT_SEED)
    print(f"solute partition train/val/test = {gc}")

    te_solute_set = set(te_solute_rows.tolist())
    selected = []
    for pid in np.unique(pair_id[te_solute_rows]):
        rows_pid = np.where(pair_id == pid)[0]
        assert set(rows_pid).issubset(te_solute_set), f"leakage: pair {pid} spans partitions"
        T_p = T_arr[rows_pid]
        if len(T_p) < 3:
            continue
        T_min, T_max = float(T_p.min()), float(T_p.max())
        if (T_max - T_min) < 20.0:
            continue
        upper = rows_pid[T_p > T_min + 0.75 * (T_max - T_min)]
        if len(upper):
            selected.append(upper)

    te_idx = np.sort(np.concatenate(selected))
    np.savez_compressed(RES / "splits_joint.npz",
                        coldsol_textrap_train=tr_idx,
                        coldsol_textrap_val=va_idx,
                        coldsol_textrap_test=te_idx)
    print(f"eligible test pairs: {len(selected):,}")
    print(f"rows train/val/test = {len(tr_idx):,} / {len(va_idx):,} / {len(te_idx):,}")
    print(f"wrote {RES / 'splits_joint.npz'}")
    print("\nnotebook 08 reported: 1,106 pairs, 2,820 test rows, 81,040 / 8,806 train/val")


if __name__ == "__main__":
    main()
