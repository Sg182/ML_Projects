"""
Matched-encoder MLP models for the direct-vs-physics ablation.

All three families share:
  - identical solute + solvent RDKit descriptor input
  - identical MLP backbone (same layer widths, activations, dropout)
  - identical output loss (MSE on LogS(mol/L))

Difference is ONLY in how temperature enters the model and how logS(T) is produced.

  A. Direct       : x = [Xsol, Xsolv, T_std];        head -> 1 scalar (logS)
  D. 1/T control  : x = [Xsol, Xsolv, T_std, invT_std]; head -> 1 scalar
  B. Van't Hoff   : x = [Xsol, Xsolv];               head -> 2 scalars (a, b)
                    logS(T) = a + b * (1/T_ref - 1/T)  with T_ref = 298.15 K

The (a, b) in Model B are called EFFECTIVE thermodynamic parameters, not
ΔH_sol / ΔS_sol. Identifying them with the physical quantities requires
assumptions (ideal solution, T-independent ΔH, defined reference state)
that this experiment does not test.
"""

from __future__ import annotations

import torch
from torch import nn

T_REF = 298.15  # K, standard reference temperature (25 °C)


class SharedBackbone(nn.Module):
    """MLP backbone shared across all model families.

    Widths kept small so CPU training is tractable and the parameter budget
    is not the differentiator.
    """

    def __init__(self, in_dim: int, hidden_dims=(256, 128), dropout=0.15):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class DirectModel(nn.Module):
    """Model A / D. Concats molecular features with a T-representation vector."""

    def __init__(self, n_desc: int, n_T_features: int, hidden_dims=(256, 128), dropout=0.15):
        super().__init__()
        self.n_T_features = n_T_features
        self.backbone = SharedBackbone(n_desc + n_T_features, hidden_dims, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x_desc, x_T):
        """x_desc: (B, n_desc) already standardized; x_T: (B, n_T_features) standardized."""
        h = self.backbone(torch.cat([x_desc, x_T], dim=1))
        return self.head(h).squeeze(-1)


class VantHoffModel(nn.Module):
    """Model B. Temperature enters ONLY through the analytical composition, never the encoder."""

    def __init__(self, n_desc: int, hidden_dims=(256, 128), dropout=0.15, T_ref: float = T_REF):
        super().__init__()
        self.backbone = SharedBackbone(n_desc, hidden_dims, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 2)  # (a, b)
        self.T_ref = T_ref

    def forward(self, x_desc, T_raw):
        """x_desc: (B, n_desc) standardized; T_raw: (B,) in Kelvin (NOT standardized)."""
        h = self.backbone(x_desc)
        ab = self.head(h)  # (B, 2)
        a = ab[:, 0]
        b = ab[:, 1]
        inv_T = 1.0 / T_raw
        inv_T_ref = 1.0 / self.T_ref
        return a + b * (inv_T_ref - inv_T)


def build_model(kind: str, n_desc: int, n_T_features: int = 1, **kwargs) -> nn.Module:
    """Factory."""
    if kind == "A":
        return DirectModel(n_desc=n_desc, n_T_features=1, **kwargs)
    if kind == "D":
        return DirectModel(n_desc=n_desc, n_T_features=2, **kwargs)
    if kind == "B":
        return VantHoffModel(n_desc=n_desc, **kwargs)
    raise ValueError(f"Unknown model kind {kind!r}")
