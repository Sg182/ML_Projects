"""
GINE encoder and the two graph-based model heads for notebook 10.

The heads mirror scripts/models.py exactly, so the only thing that changes
between the MLP and GINE arms of the 2x2 is how chemistry is represented:

    A_GINE  : [h_solute, h_solvent, T_std]  -> MLP -> logS          (direct)
    Bc_GINE : [h_solute, h_solvent]         -> MLP -> (a, beta)
              logS(T) = a + beta * b_scale * (1/T_ref - 1/T)        (Van't Hoff)

Solute and solvent get separate encoders: this dataset has 1,445 solutes but
only 70 solvents, so they occupy very different chemical spaces and sharing
weights would be the wrong prior.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

T_REF = 298.15


class GINEEncoder(nn.Module):
    """Small GINE stack: node embedding -> n_layers x GINEConv -> mean pool."""

    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=3, dropout=0.15):
        super().__init__()
        self.node_lin = nn.Linear(node_dim, hidden)
        self.edge_lin = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, batch):
        h = self.node_lin(batch.x)
        e = self.edge_lin(batch.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, batch.edge_index, e))))
        return global_mean_pool(h, batch.batch)


class _Head(nn.Module):
    """Shared MLP head on top of [h_solute, h_solvent] (+ optional T features)."""

    def __init__(self, in_dim, out_dim, hidden_dims=(256, 128), dropout=0.15):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DirectGINE(nn.Module):
    """A_GINE — temperature appended as a standardized feature."""

    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=3, dropout=0.15,
                 head_dims=(256, 128)):
        super().__init__()
        self.enc_solute = GINEEncoder(node_dim, edge_dim, hidden, n_layers, dropout)
        self.enc_solvent = GINEEncoder(node_dim, edge_dim, hidden, n_layers, dropout)
        self.head = _Head(2 * hidden + 1, 1, head_dims, dropout)

    def forward(self, sol_batch, solv_batch, sol_idx, solv_idx, T_feat):
        h_s = self.enc_solute(sol_batch)[sol_idx]
        h_v = self.enc_solvent(solv_batch)[solv_idx]
        return self.head(torch.cat([h_s, h_v, T_feat.unsqueeze(1)], dim=1)).squeeze(-1)


class VantHoffGINE(nn.Module):
    """Bc_GINE — same conditioned Van't Hoff composition as models.py's Bc."""

    def __init__(self, node_dim, edge_dim, b_scale, hidden=128, n_layers=3, dropout=0.15,
                 head_dims=(256, 128), T_ref=T_REF):
        super().__init__()
        self.enc_solute = GINEEncoder(node_dim, edge_dim, hidden, n_layers, dropout)
        self.enc_solvent = GINEEncoder(node_dim, edge_dim, hidden, n_layers, dropout)
        self.head = _Head(2 * hidden, 2, head_dims, dropout)
        self.T_ref = T_ref
        self.register_buffer("b_scale", torch.tensor(float(b_scale)))

    def ab(self, sol_batch, solv_batch, sol_idx, solv_idx):
        """Return (a, b) with b already on the physical scale (model convention)."""
        h_s = self.enc_solute(sol_batch)[sol_idx]
        h_v = self.enc_solvent(solv_batch)[solv_idx]
        out = self.head(torch.cat([h_s, h_v], dim=1))
        return out[:, 0], out[:, 1] * self.b_scale

    def forward(self, sol_batch, solv_batch, sol_idx, solv_idx, T_raw):
        a, b = self.ab(sol_batch, solv_batch, sol_idx, solv_idx)
        return a + b * (1.0 / self.T_ref - 1.0 / T_raw)


def build_gine(kind, node_dim, edge_dim, b_scale=None, **kw):
    if kind == "A_GINE":
        return DirectGINE(node_dim, edge_dim, **kw)
    if kind == "Bc_GINE":
        if b_scale is None:
            raise ValueError("Bc_GINE requires b_scale")
        return VantHoffGINE(node_dim, edge_dim, b_scale, **kw)
    raise ValueError(f"unknown kind {kind!r}")
