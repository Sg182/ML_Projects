"""
Re-run the descriptor-MLP arm (A, Bc) inside the PyG environment.

Notebook 10's 2x2 crosses representation (RDKit MLP vs GINE) with head (direct
vs Van't Hoff). PyTorch Geometric only exists in the `ml` env (torch 2.13),
while notebooks 06-09 ran on torch 2.7.1. results/parity_report.md shows early
stopping is a discrete choice, so a version change can move RMSE far more than
float noise. Comparing a GINE run here against an MLP run there would confound
representation with environment, so the MLP arm is reproduced here.

These numbers are for notebook 10's internal comparison only; notebook 06
remains the reference for the MLP results reported elsewhere.

Usage:  <ml-env>/bin/python scripts/run_mlp_pygenv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn  # noqa: F401  - before torch (libomp)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import train_one, metrics_row  # noqa: E402
from prepare_b_scale import per_pair_slopes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "metrics_mlp_pygenv.csv"
PREDS = RES / "preds_mlp_pygenv"
SPLITS = ["random", "textrap", "coldsol", "coldpair", "coldsol_textrap"]
SEEDS = [42, 123, 456]


def joint_indices():
    d = np.load(RES / "splits_joint.npz")
    return (d["coldsol_textrap_train"], d["coldsol_textrap_val"], d["coldsol_textrap_test"])


def main():
    PREDS.mkdir(parents=True, exist_ok=True)
    b_scales = json.loads((RES / "b_scale.json").read_text())["splits"]

    feats = np.load(RES / "features.npz", allow_pickle=True)
    tr_j = joint_indices()[0]
    sl, _, _ = per_pair_slopes(feats["T"], feats["y"], feats["pair_id"], tr_j)
    b_scales["coldsol_textrap"] = {"b_scale": float(np.median(np.abs(sl)))}

    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = {(str(r.model), str(r.split), int(r.seed)) for r in d.itertuples()}

    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    for split in SPLITS:
        bs = b_scales[split]["b_scale"]
        idx = joint_indices() if split == "coldsol_textrap" else None
        print(f"\n=== {split}  (b_scale = {bs:.1f} K) ===", flush=True)
        for kind in ["A", "Bc"]:
            label = f"{kind}_MLP"
            for seed in SEEDS:
                if (label, split, seed) in done:
                    print(f"  [cached] {label} seed {seed}", flush=True)
                    continue
                metrics, elapsed, preds = train_one(
                    kind, split, seed, device="cpu",
                    b_scale=(bs if kind == "Bc" else None),
                    verbose=False, split_indices=idx)
                row = metrics_row(label, split, seed, metrics, elapsed,
                                  extra={"b_scale": bs if kind == "Bc" else None})
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT, index=False)
                preds["model"] = kind          # keep the notebook-07 naming for preds
                np.savez_compressed(PREDS / f"preds_{kind}_{split}_seed{seed}.npz", **preds)
                print(f"  {label:>6} seed {seed}: test_rmse={metrics['test']['rmse']:.4f}  "
                      f"({elapsed:.0f}s)", flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
