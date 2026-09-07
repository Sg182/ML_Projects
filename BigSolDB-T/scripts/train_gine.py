"""
Training loop for the GINE arm of notebook 10's 2x2.

Mirrors scripts/train.py's frozen configuration exactly (same optimiser, lr,
weight decay, batch size, early-stopping rule and head dimensions) so that the
ONLY difference between the MLP and GINE arms is how chemistry is represented.

Efficiency note: a minibatch of 512 measurements touches at most a few hundred
distinct molecules, and BigSolDB has only 1,515 in total. Each step therefore
encodes just the unique solutes and solvents present in that batch and gathers
the embeddings by index — exact, and far cheaper than building one graph per
measurement.

Requires PyTorch Geometric (the `ml` conda env).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sklearn  # noqa: F401  - import before torch in this env (libomp clash)
import torch
from torch import nn
from torch_geometric.data import Batch

from gine import build_gine

# This box has 12 cores; torch defaults to 6. GINE is compute-bound.
torch.set_num_threads(max(1, min(10, torch.get_num_threads() * 2)))

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# --- frozen configuration, matched to scripts/train.py ---------------------
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
GRAD_CLIP = 1.0
HEAD_DIMS = (256, 128)
DROPOUT = 0.15
# GINE-specific (kept small on purpose)
GNN_HIDDEN = 128
GNN_LAYERS = 3
T_REF = 298.15
# ---------------------------------------------------------------------------


def load_graph_data():
    g = torch.load(RES / "graphs.pt", weights_only=False)
    feats = np.load(RES / "features.npz", allow_pickle=True)
    return g, feats["T"].astype(np.float64), feats["y"].astype(np.float64), feats["pair_id"]


def _score(y_pred, y_true):
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": int(len(y_true))}


class _Encoder:
    """Collates and caches the per-batch graph batches."""

    def __init__(self, g):
        self.sol = g["solute_graphs"]
        self.solv = g["solvent_graphs"]
        self.sol_id = g["solute_id"]
        self.solv_id = g["solvent_id"]

    def batch_for(self, rows):
        s_ids = self.sol_id[rows]
        v_ids = self.solv_id[rows]
        us, s_inv = torch.unique(s_ids, return_inverse=True)
        uv, v_inv = torch.unique(v_ids, return_inverse=True)
        sb = Batch.from_data_list([self.sol[i] for i in us.tolist()])
        vb = Batch.from_data_list([self.solv[i] for i in uv.tolist()])
        return sb, vb, s_inv, v_inv


def train_one_gine(kind, split_name, seed, b_scale=None, split_indices=None,
                   verbose=False, max_epochs=MAX_EPOCHS):
    """Train A_GINE or Bc_GINE. Returns (metrics, elapsed, preds)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    g, T, y, pair_id = load_graph_data()
    enc = _Encoder(g)

    if split_indices is None:
        # The joint chemistry+temperature split lives in its own file because
        # notebook 08 built it inline (see scripts/make_joint_split.py).
        f = "splits_joint.npz" if split_name == "coldsol_textrap" else "splits.npz"
        sp = np.load(RES / f)
        tr, va, te = sp[f"{split_name}_train"], sp[f"{split_name}_val"], sp[f"{split_name}_test"]
    else:
        tr, va, te = split_indices

    # Temperature feature for the direct model: standardized on TRAIN rows only.
    t_mean, t_std = T[tr].mean(), T[tr].std()
    T_std = ((T - t_mean) / t_std).astype(np.float32)
    T_raw = T.astype(np.float32)

    model = build_gine(kind, g["node_dim"], g["edge_dim"], b_scale=b_scale,
                       hidden=GNN_HIDDEN, n_layers=GNN_LAYERS, dropout=DROPOUT,
                       head_dims=HEAD_DIMS)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    y_t = torch.from_numpy(y.astype(np.float32))
    feat = torch.from_numpy(T_raw if kind == "Bc_GINE" else T_std)

    def run_eval(rows):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), BATCH_SIZE):
                chunk = torch.from_numpy(rows[i:i + BATCH_SIZE])
                sb, vb, si, vi = enc.batch_for(chunk)
                preds.append(model(sb, vb, si, vi, feat[chunk]).numpy())
        return np.concatenate(preds)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), BATCH_SIZE):
            chunk = torch.from_numpy(tr[perm[i:i + BATCH_SIZE]])
            sb, vb, si, vi = enc.batch_for(chunk)
            loss = crit(model(sb, vb, si, vi, feat[chunk]), y_t[chunk])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
        v = _score(run_eval(va), y[va])["rmse"]
        if v < best - EARLY_STOP_MIN_DELTA:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"    epoch {epoch:3d}  val={v:.4f}  best={best:.4f}  bad={bad}", flush=True)
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    metrics = {}
    for phase, rows in (("train", tr), ("val", va), ("test", te)):
        p = run_eval(rows)
        metrics[phase] = _score(p, y[rows])
        if phase == "test":
            test_pred = p

    preds = {"model": kind, "split": split_name, "seed": seed,
             "test_idx": np.asarray(te), "pair_id": pair_id[te],
             "T": T[te], "y_true": y[te], "y_pred": test_pred.astype(np.float64)}

    # For the physics head, store the per-row (a, b) so notebook 10 can compare
    # them against empirical Van't Hoff parameters.
    if kind == "Bc_GINE":
        model.eval()
        A, B = [], []
        with torch.no_grad():
            for i in range(0, len(te), BATCH_SIZE):
                chunk = torch.from_numpy(np.asarray(te)[i:i + BATCH_SIZE])
                sb, vb, si, vi = enc.batch_for(chunk)
                a, b = model.ab(sb, vb, si, vi)
                A.append(a.numpy()); B.append(b.numpy())
        preds["slope_a"] = np.concatenate(A).astype(np.float64)
        preds["slope_b"] = np.concatenate(B).astype(np.float64)
        preds["b_scale"] = float(b_scale)

    return metrics, elapsed, preds
