"""
Molecular graphs for the GINE representation study (notebook 10).

BigSolDB has 100,983 measurements but only ~1,515 distinct molecules (1,445
solutes + 70 solvents). Encoding a graph per measurement would redo the same
work thousands of times, so molecules are featurised ONCE here and cached; the
training loop then encodes only the unique molecules present in a minibatch and
gathers the resulting embeddings by index.

Node and edge features are deliberately plain — the point of notebook 10 is to
test whether a learned representation helps, not to engineer graph features.

Requires RDKit and PyTorch Geometric (the `ml` conda env).
Run:  <ml-env>/bin/python scripts/graphs.py   ->  results/graphs.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Elements covering essentially all of BigSolDB; anything else lands in "other".
ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

NODE_DIM = len(ELEMENTS) + 1 + 6 + 5 + 5 + len(HYBRIDIZATIONS) + 1 + 1 + 1
EDGE_DIM = len(BOND_TYPES) + 1 + 3   # bond-type one-hot (+other), conjugated, in-ring, pad


def _onehot(value, choices):
    v = [0.0] * (len(choices) + 1)
    v[choices.index(value) if value in choices else len(choices)] = 1.0
    return v


def atom_features(atom):
    return (
        _onehot(atom.GetAtomicNum(), ELEMENTS)
        + _onehot(atom.GetDegree(), [0, 1, 2, 3, 4])
        + _onehot(atom.GetFormalCharge(), [-2, -1, 0, 1])
        + _onehot(atom.GetTotalNumHs(), [0, 1, 2, 3])
        + _onehot(atom.GetHybridization(), HYBRIDIZATIONS)[:-1]
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() / 100.0]
    )


def bond_features(bond):
    return (
        _onehot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing()), 0.0]
    )


def mol_to_data(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]                      # undirected -> both directions
        dst += [j, i]
        eattr += [f, f]
    if not src:                            # single-atom molecule: self loop
        src, dst = [0], [0]
        eattr = [[0.0] * EDGE_DIM]

    return Data(x=x,
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=torch.tensor(eattr, dtype=torch.float))


def main():
    df = pd.read_csv(ROOT / "data" / "BigSolDBv2.0.csv").dropna(
        subset=["LogS(mol/L)"]).reset_index(drop=True)
    solute = df["SMILES_Solute"].astype(str).to_numpy()
    solvent = df["SMILES_Solvent"].astype(str).to_numpy()

    out = {}
    for name, arr in (("solute", solute), ("solvent", solvent)):
        uniq = list(dict.fromkeys(arr.tolist()))
        graphs, index = [], {}
        for s in uniq:
            d = mol_to_data(s)
            if d is None:
                raise ValueError(f"RDKit could not parse {name} SMILES: {s!r}")
            index[s] = len(graphs)
            graphs.append(d)
        out[f"{name}_graphs"] = graphs
        out[f"{name}_id"] = torch.tensor([index[s] for s in arr], dtype=torch.long)
        n_atoms = [g.num_nodes for g in graphs]
        print(f"{name:<8} {len(graphs):5,} unique molecules   "
              f"atoms: min {min(n_atoms)}, median {int(np.median(n_atoms))}, max {max(n_atoms)}")

    out["node_dim"] = NODE_DIM
    out["edge_dim"] = EDGE_DIM
    torch.save(out, RES / "graphs.pt")
    print(f"\nnode_dim={NODE_DIM}  edge_dim={EDGE_DIM}")
    print(f"wrote {RES / 'graphs.pt'}")


if __name__ == "__main__":
    main()
