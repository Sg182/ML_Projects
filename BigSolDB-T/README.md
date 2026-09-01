# BigSolDB-T — Thermodynamic Inductive Bias for Solubility Extrapolation

**Publication question:** Does thermodynamic inductive bias improve solubility prediction specifically under temperature and chemical distribution shift?

**Scope:** narrow methodology study on BigSolDB 2.0 comparing four matched-encoder models. Not a SOTA architecture race.

**Portfolio value:** clean, reproducible, interview-ready. Does not depend on the publication succeeding.

## Model families (all with identical MLP backbone, matched features)

| Model | T representation | Head | logS(T) prediction |
|---|---|---|---|
| **A. Direct** | Raw `T` appended | MLP → 1 | Predicted directly |
| **D. 1/T control** | `[T, 1/T]` appended | MLP → 1 | Predicted directly |
| **B. Van't Hoff physics** | None (T not in encoder input) | MLP → (a, b) | Composed: `logS(T) = a + b · (1/T_ref − 1/T)` where T_ref = 298.15 K |
| **C. Kirchhoff (deferred)** | | | Not implemented until A/D/B evaluated |

- **A vs D**: does representing T as 1/T explain any Van't Hoff advantage?
- **A vs B**: does explicit thermodynamic composition help?
- **D vs B**: is B better than a direct model that already has 1/T as a feature?

**Parameters a, b in Model B are called "effective thermodynamic parameters"**, not ΔH_sol / ΔS_sol, because their identification with the true physical quantities requires assumptions (ideal solution, T-independent ΔH, defined reference state) that are not tested here.

## Features (Stage 1 MVP — no GNN)

- 199 RDKit descriptors for solute
- 199 RDKit descriptors for solvent (same descriptor set, computed on solvent SMILES)
- Temperature representation per model above
- `StandardScaler` fitted on train only

## Splits (Stage 1 MVP)

- **Split 1 — Random measurement split:** 80/10/10 stratified by nothing; measurement-level.
- **Split 2 — T-extrapolation split:** for each pair with ≥5 measurements and ΔT ≥ 20 K, hold out the highest 25 % of temperature measurements as test; remaining lower-T measurements go into train/val by random 90/10. Pairs not meeting the eligibility criteria are placed entirely in train (they cannot contribute a T-extrapolation signal but still provide broad-coverage training data).

Other splits (cold-solute, cold-solvent, cold-pair, scaffold) are deferred until A/D/B results on the two initial splits are reviewed.

## Directory layout

```
BigSolDB-T/
├── README.md                       (this file)
├── data/
│   ├── BigSolDBv2.0.csv            (downloaded from Zenodo, unmodified)
│   └── BigSolDBv2.0_densities.csv
├── notebooks/
│   ├── 01_audit.ipynb              (data audit)
│   ├── 02_prepare_features.ipynb   (RDKit descriptors + clean + save)
│   ├── 03_make_splits.ipynb        (5 splits with leakage audit)
│   ├── 04_train.ipynb              (models A/D/B with shared MLP)
│   └── 05_summarize.ipynb          (ΔRMSE comparison tables)
├── scripts/
│   └── models.py                   (model classes; imported by 04_train.ipynb)
└── results/
    ├── audit_report.md
    ├── features.npz                (cached descriptors)
    ├── splits.npz                  (cached train/val/test indices)
    ├── metrics.csv                 (aggregate metrics per (model, split, seed))
    ├── comparison_stage2.md        (final ΔRMSE table)
    └── run_log.md                  (results narrative)
```

**Execution order:** run notebooks 01 → 02 → 03 → 04 → 05.
The training notebook's parameter cell (`COMBOS`, `SEED`) controls which
(model, split) combinations are run.

## Reproducibility

- Python 3.x, PyTorch 2.x, scikit-learn 1.9, RDKit 2026.03.
- Seeds: {42} during debugging; {42, 123, 456} for reported comparison; escalate to five only for final experiments.
- All CSV / numpy artifacts are cached so re-running is fast.
