"""
Repeated cold-solute holdouts for the MLP-vs-GINE representation comparison
(notebook 11).

Notebook 08 established that a single cold-solute partition can be misleading:
a within-partition bootstrap looked decisive and repeated group holdouts
reversed the sign. Notebook 10's GINE cold-solute penalty rests on exactly one
partition, so it gets the same treatment here.

Partitions are the SAME five used in notebook 08 (`group_split` over solutes with
split seeds 11/22/33/44/55) — no new split protocol.

All four cells of the 2x2 are trained on identical rows for every partition:
    A_MLP, Bc_MLP    (RDKit descriptors, scripts/train.py)
    A_GINE, Bc_GINE  (molecular graphs, scripts/train_gine.py)

Model seeds are reduced to 2 (42, 123) rather than 3: GINE costs ~10 min/run and
the partition, not the seed, is the unit of replication here. Seed spread on this
split was ~0.018 against an effect of ~0.084.

Resumable — re-launch to continue. Requires the PyG env.
Usage:  <ml-env>/bin/python scripts/run_coldsol_repeat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn  # noqa: F401  - before torch (libomp)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import train_one                      # noqa: E402  descriptor MLP
from train_gine import train_one_gine            # noqa: E402  graph encoder
from prepare_b_scale import per_pair_slopes      # noqa: E402
from make_joint_split import group_split         # noqa: E402  same helper as nb08

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "metrics_coldsol_repeat.csv"
PREDS = RES / "preds_coldsol_repeat"

SPLIT_SEEDS = [11, 22, 33, 44, 55]     # identical to notebook 08
MODEL_SEEDS = [42, 123]
MODELS = ["A_MLP", "Bc_MLP", "A_GINE", "Bc_GINE"]


def main():
    PREDS.mkdir(parents=True, exist_ok=True)
    feats = np.load(RES / "features.npz", allow_pickle=True)
    T, y, pair_id = feats["T"], feats["y"], feats["pair_id"]
    df = pd.read_csv(ROOT / "data" / "BigSolDBv2.0.csv").dropna(
        subset=["LogS(mol/L)"]).reset_index(drop=True)
    solute_id, uniq = pd.factorize(df["SMILES_Solute"].astype(str))

    done = set()
    rows = []
    if OUT.exists():
        d = pd.read_csv(OUT)
        rows = d.to_dict("records")
        done = {(str(r.model), int(r.split_seed), int(r.model_seed)) for r in d.itertuples()}

    for ss in SPLIT_SEEDS:
        tr, va, te, gc = group_split(solute_id, len(uniq), seed=ss)
        slopes, _, _ = per_pair_slopes(T, y, pair_id, tr)
        b_scale = float(np.median(np.abs(slopes)))
        print(f"\n=== cold-solute split_seed={ss}  solutes tr/va/te={gc}  "
              f"rows {len(tr)}/{len(va)}/{len(te)}  b_scale={b_scale:.1f} K ===", flush=True)

        for kind in MODELS:
            for ms in MODEL_SEEDS:
                if (kind, ss, ms) in done:
                    print(f"  [cached] {kind} seed {ms}", flush=True)
                    continue
                is_phys = kind.startswith("Bc")
                bs = b_scale if is_phys else None
                if kind.endswith("GINE"):
                    metrics, elapsed, preds = train_one_gine(
                        kind, f"coldsol_ss{ss}", ms, b_scale=bs,
                        split_indices=(tr, va, te))
                else:
                    metrics, elapsed, preds = train_one(
                        "Bc" if is_phys else "A", f"coldsol_ss{ss}", ms,
                        device="cpu", b_scale=bs, split_indices=(tr, va, te))
                row = {"model": kind, "split_seed": ss, "model_seed": ms,
                       "b_scale": bs, "n_solutes_test": gc[2],
                       "train_n": len(tr), "val_n": len(va), "test_n": len(te),
                       "test_rmse": metrics["test"]["rmse"],
                       "test_mae": metrics["test"]["mae"],
                       "test_r2": metrics["test"]["r2"],
                       "elapsed_s": round(elapsed, 1)}
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT, index=False)
                np.savez_compressed(
                    PREDS / f"preds_{kind}_ss{ss}_seed{ms}.npz", **preds)
                print(f"  {kind:>8} seed {ms}: test_rmse={metrics['test']['rmse']:.4f}  "
                      f"({elapsed/60:.1f} min)", flush=True)
    print("\nALL COLDSOL REPEATS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
