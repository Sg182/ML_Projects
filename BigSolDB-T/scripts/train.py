"""
Reusable training loop for the A / D / B / Bc comparison.

Single source of truth for the frozen training configuration. Notebooks 06 and
08 import from here rather than defining their own loop, so every reported run
shares one code path.

Public API (as used by notebooks 06 and 08):

    RES, HIDDEN, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE,
    MAX_EPOCHS, PATIENCE, EARLY_STOP_MIN_DELTA, GRAD_CLIP, SPLIT_NAMES
    load_features()  -> (X_desc, T, y, pair_id)
    load_split(name) -> (train_idx, val_idx, test_idx)
    train_one(kind, split, seed, device="cpu", b_scale=None, verbose=False,
              split_indices=None) -> (metrics, elapsed, preds)
    metrics_row(kind, split, seed, metrics, elapsed, extra=None) -> dict
    save_preds(preds, out_dir) -> Path

`split_indices` lets callers supply an ad-hoc partition (notebook 08's repeated
group holdouts and the joint chemistry+temperature split); `split` is then only
a label used for filenames and bookkeeping.

Training is bit-deterministic on a fixed machine but NOT across platforms.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler  # import before torch (libomp)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# --- frozen configuration (do not override in notebooks) -------------------
HIDDEN = (256, 128)
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
GRAD_CLIP = 1.0
# ---------------------------------------------------------------------------

SPLIT_NAMES = ("random", "textrap", "coldsol", "coldpair", "coldsolv")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features():
    """(X_desc, T, y, pair_id). X_desc is solute descriptors then solvent."""
    d = np.load(RES / "features.npz", allow_pickle=True)
    X_desc = np.concatenate([d["X_sol"], d["X_solv"]], axis=1).astype(np.float32)
    return X_desc, d["T"].astype(np.float32), d["y"].astype(np.float32), d["pair_id"]


def load_split(name: str):
    d = np.load(RES / "splits.npz")
    return d[f"{name}_train"], d[f"{name}_val"], d[f"{name}_test"]


def build_T_features(kind: str, T: np.ndarray):
    """A sees T; D sees [T, 1/T]; B and Bc take T through the Van't Hoff
    composition instead and so get no appended temperature feature."""
    if kind == "A":
        return T.reshape(-1, 1).astype(np.float32)
    if kind == "D":
        return np.stack([T, 1.0 / T], axis=1).astype(np.float32)
    if kind in ("B", "Bc"):
        return None
    raise ValueError(f"unknown model kind {kind!r}")


def _loader(x_desc, second, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x_desc), torch.from_numpy(second), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def _predict(model, loader, device):
    model.eval()
    p, t = [], []
    with torch.no_grad():
        for x, second, y in loader:
            x, second = x.to(device), second.to(device)
            p.append(model(x, second).detach().cpu().numpy())
            t.append(y.numpy())
    return np.concatenate(p), np.concatenate(t)


def _score(y_pred, y_true) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def _head_ab(model, x_desc, device, batch=4096):
    """Recover (a, b_m) from a 2-unit Van't Hoff head, in model convention.

    For Bc the head emits beta, so the physical slope is beta * b_scale; for
    stock B the head emits b directly. Both are returned as b_m, so downstream
    analysis uses one convention regardless of parameterization.
    """
    model.eval()
    A, B = [], []
    scale = float(model.b_scale) if hasattr(model, "b_scale") else 1.0
    with torch.no_grad():
        for i in range(0, len(x_desc), batch):
            xb = torch.from_numpy(x_desc[i:i + batch]).to(device)
            ab = model.head(model.backbone(xb)).cpu().numpy()
            A.append(ab[:, 0]); B.append(ab[:, 1] * scale)
    return np.concatenate(A), np.concatenate(B)


def train_one(kind: str, split: str, seed: int, device: str = "cpu",
              b_scale: float | None = None, verbose: bool = False,
              split_indices=None):
    """Train one (model, split, seed) and return (metrics, elapsed, preds).

    `preds` is a dict ready for save_preds: it carries the identifying labels
    plus per-test-row arrays, and (a, b) for the two-parameter heads.
    """
    if kind == "Bc" and b_scale is None:
        raise ValueError("model Bc requires b_scale (see scripts/prepare_b_scale.py)")

    set_seed(seed)
    X_desc, T, y, pair_id = load_features()
    tr, va, te = load_split(split) if split_indices is None else split_indices

    scaler = StandardScaler().fit(X_desc[tr])
    Xs = scaler.transform(X_desc).astype(np.float32)

    T_feat = build_T_features(kind, T)
    if T_feat is not None:
        second = StandardScaler().fit(T_feat[tr]).transform(T_feat).astype(np.float32)
    else:
        second = T.astype(np.float32)          # raw Kelvin for the physics head

    kwargs = {"b_scale": b_scale} if kind == "Bc" else {}
    model = build_model(kind, n_desc=Xs.shape[1], hidden_dims=HIDDEN,
                        dropout=DROPOUT, **kwargs).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    tl = _loader(Xs[tr], second[tr], y[tr], shuffle=True)
    vl = _loader(Xs[va], second[va], y[va], shuffle=False)
    el = _loader(Xs[te], second[te], y[te], shuffle=False)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, s, yb in tl:
            x, s, yb = x.to(device), s.to(device), yb.to(device)
            loss = criterion(model(x, s), yb)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
        v = _score(*_predict(model, vl, device))
        if v["rmse"] < best - EARLY_STOP_MIN_DELTA:
            best, bad = v["rmse"], 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (epoch % 10 == 0):
            print(f"    epoch {epoch:3d}  val_rmse={v['rmse']:.4f}  best={best:.4f}")
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    metrics = {}
    for phase, loader in (("train", tl), ("val", vl), ("test", el)):
        yp, yt = _predict(model, loader, device)
        metrics[phase] = _score(yp, yt)
        if phase == "test":
            test_pred, test_true = yp, yt

    # Test loader is unshuffled, so rows align with te.
    preds = {
        "model": kind, "split": split, "seed": seed,
        "test_idx": np.asarray(te), "pair_id": pair_id[te],
        "T": T[te].astype(np.float64),
        "y_true": test_true.astype(np.float64),
        "y_pred": test_pred.astype(np.float64),
    }
    if kind in ("B", "Bc"):
        a_all, b_all = _head_ab(model, Xs[te], device)
        preds["slope_a"] = a_all.astype(np.float64)
        preds["slope_b"] = b_all.astype(np.float64)
        preds["b_scale"] = float(b_scale) if b_scale is not None else 1.0

    return metrics, elapsed, preds


def metrics_row(kind, split, seed, metrics, elapsed, extra=None) -> dict:
    row = {"model": kind, "split": split, "seed": int(seed),
           "elapsed_s": round(float(elapsed), 1)}
    for phase in ("train", "val", "test"):
        for k in ("rmse", "mae", "r2", "n"):
            row[f"{phase}_{k}"] = metrics[phase][k]
    if extra:
        row.update(extra)
    return row


def save_preds(preds: dict, out_dir: Path) -> Path:
    """Write results/<out_dir>/preds_{model}_{split}_seed{seed}.npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"preds_{preds['model']}_{preds['split']}_seed{preds['seed']}.npz"
    np.savez_compressed(path, **preds)
    return path
