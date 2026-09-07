"""
Resumable GINE sweep for notebook 10.

Runs A_GINE / Bc_GINE over the requested splits and seeds, appending one row per
run to results/metrics_gine.csv and saving per-run predictions. Resumable: a
(model, split, seed) already present in the CSV is skipped, so an interrupted
sweep can simply be re-launched.

Usage:  <ml-env>/bin/python scripts/run_gine.py random
        <ml-env>/bin/python scripts/run_gine.py textrap coldsol coldpair
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
from train_gine import train_one_gine

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
METRICS_DIR = RES / "gine_runs"   # one CSV per split: concurrent sweeps cannot race
PREDS = RES / "preds_gine"
SEEDS = [42, 123, 456]
MODELS = ["A_GINE", "Bc_GINE"]


def csv_for(split):
    return METRICS_DIR / f"metrics_gine_{split}.csv"


def load_done(split):
    f = csv_for(split)
    if not f.exists():
        return set()
    d = pd.read_csv(f)
    return {(str(r.model), str(r.split), int(r.seed)) for r in d.itertuples()}


def append_row(split, row):
    f = csv_for(split)
    df = pd.DataFrame([row])
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)


def main(splits):
    PREDS.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    b_scales = json.loads((RES / "b_scale.json").read_text())["splits"]
    if "coldsol_textrap" not in b_scales:
        # Same rule as everywhere else: median |b| over TRAINING rows only.
        from prepare_b_scale import per_pair_slopes
        feats = np.load(RES / "features.npz", allow_pickle=True)
        tr = np.load(RES / "splits_joint.npz")["coldsol_textrap_train"]
        sl, _, _ = per_pair_slopes(feats["T"], feats["y"], feats["pair_id"], tr)
        b_scales["coldsol_textrap"] = {"b_scale": float(np.median(np.abs(sl)))}
    for split in splits:
        done = load_done(split)
        bs = b_scales[split]["b_scale"]
        print(f"\n=== {split}  (b_scale = {bs:.1f} K) ===", flush=True)
        for kind in MODELS:
            for seed in SEEDS:
                if (kind, split, seed) in done:
                    print(f"  [cached] {kind} seed {seed}", flush=True)
                    continue
                metrics, elapsed, preds = train_one_gine(
                    kind, split, seed,
                    b_scale=(bs if kind == "Bc_GINE" else None))
                row = {"model": kind, "split": split, "seed": seed,
                       "b_scale": bs if kind == "Bc_GINE" else None,
                       "elapsed_s": round(elapsed, 1)}
                for ph in ("train", "val", "test"):
                    for k in ("rmse", "mae", "r2", "n"):
                        row[f"{ph}_{k}"] = metrics[ph][k]
                append_row(split, row)
                np.savez_compressed(PREDS / f"preds_{kind}_{split}_seed{seed}.npz", **preds)
                print(f"  {kind:>8} seed {seed}: test_rmse={metrics['test']['rmse']:.4f}  "
                      f"r2={metrics['test']['r2']:.4f}  ({elapsed/60:.1f} min)", flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["random"])
