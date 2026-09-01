# BigSolDB 2.0 — Audit Report

- File: `/home/sg182/ML_Projects/BigSolDB-T/data/BigSolDBv2.0.csv`
- Rows: 103,944

## Column integrity
- All expected columns present.

## Unique counts
- Unique solutes (by `SMILES_Solute` raw string): 1,448
- Unique solvents (by `SMILES_Solvent` raw string): 209
- Unique (solute, solvent) pairs: 11,255
- Unique source articles: 1,595
- FDA-approved solute rows: 24,483

**Note on unique counts:** raw SMILES strings not standardized here — the numbers may drop slightly after RDKit canonicalization in the feature-prep step. This audit reflects the file as delivered.

## Missing values
- `CAS`: 7,456 missing
- `Solubility(mol/L)`: 2,961 missing
- `LogS(mol/L)`: 2,961 missing
- `PubChem_CID`: 2,593 missing

## Duplicates
- Full-row duplicates: 0
- Duplicates on (SMILES_Solute, SMILES_Solvent, Temperature_K): 3,601
  (these are legitimate replicate measurements; they define our aleatoric floor)

## Target distribution — `LogS(mol/L)`
- min: -9.1289
- max: +2.4911
- mean: -1.0300
- std: +1.2269
- median: -0.8940
- n_null: 2,961

## Temperature distribution — `Temperature_K`
- min: 243.150
- max: 425.770
- mean: 303.630
- std: 15.749
- median: 303.150
- n_null: 0
- n_unique: 3,331

## Measurements per (solute, solvent) pair
- Total pairs: 11,255
| Threshold | Pairs with ≥ threshold measurements |
|---|---|
| ≥1 | 11,255 |
| ≥2 | 10,851 |
| ≥3 | 10,831 |
| ≥5 | 10,730 |
| ≥10 | 4,448 |
| ≥20 | 226 |

- Median measurements/pair: 9
- Mean measurements/pair: 9.24
- Max measurements/pair: 50

## Temperature span (ΔT = Tmax − Tmin) per (solute, solvent) pair
- Pairs with only one T (ΔT=0): 409
| Threshold | Pairs with ΔT ≥ threshold |
|---|---|
| ≥10 K | 10,831 |
| ≥20 K | 10,676 |
| ≥40 K | 7,441 |
| ≥60 K | 238 |
| ≥80 K | 45 |

- Median ΔT among multi-T pairs: 40.0 K
- Max ΔT: 132.6 K

## Pairs meeting simultaneous conditions (T-extrapolation feasibility)
| Rule | Count |
|---|---|
| n_pairs_meas>=3_dT>=20K | 10,675 |
| n_pairs_meas>=5_dT>=20K | 10,617 |
| n_pairs_meas>=5_dT>=40K | 7,440 |
| n_pairs_meas>=10_dT>=40K | 3,794 |
| n_pairs_meas>=10_dT>=60K | 188 |

## Top-15 solvents by number of measurements
| Solvent SMILES | Count |
|---|---|
| `CCO` | 10,271 |
| `CO` | 8,220 |
| `CC(C)O` | 7,298 |
| `O` | 6,814 |
| `CCOC(C)=O` | 6,802 |
| `CCCO` | 6,616 |
| `CC(C)=O` | 6,062 |
| `CCCCO` | 5,613 |
| `CC#N` | 5,251 |
| `CN(C)C=O` | 2,767 |
| `Cc1ccccc1` | 2,618 |
| `CC(C)CO` | 2,440 |
| `C1COCCO1` | 2,183 |
| `COC(C)=O` | 1,908 |
| `C1CCOC1` | 1,524 |

## Top-10 solutes by number of measurements
| Solute SMILES | Count |
|---|---|
| `CN1CCN(C2=Nc3cc(Cl)ccc3Nc3ccccc32)CC1` | 416 |
| `C[C@@H]1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CC=C5C[C@@H](O)CC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C` | 380 |
| `Cn1c(=O)c2c(ncn2CC2OCCO2)n(C)c1=O` | 357 |
| `CC(=O)Nc1ccc(OC(=O)c2ccccc2OC(C)=O)cc1` | 313 |
| `O=C(O)c1ccccc1` | 311 |
| `O=[N+]([O-])N1C2C3N([N+](=O)[O-])C1C1N([N+](=O)[O-])C(C(N1[N+](=O)[O-])N3[N+](=O)[O-])N2[N+](=O)[O-]` | 295 |
| `CC(C)C(=O)Nc1ccc([N+](=O)[O-])c(C(F)(F)F)c1` | 272 |
| `Nc1ccc(S(N)(=O)=O)cc1` | 267 |
| `O=c1[nH]cc(F)c(=O)[nH]1` | 258 |
| `Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2` | 252 |

## Summary implications
- **Aleatoric floor (from Nature Sci Data paper):** RMSE ≈ 0.39 logS.
- **T-extrapolation feasibility:** driven by the `pairs with ≥N meas AND ΔT ≥ M K` rows above.
- **Sanity check on primary claims:** row count should be ≈ 103,944; solutes ≈ 1,448; solvents ≈ 213 per Krasnov et al. 2025.
