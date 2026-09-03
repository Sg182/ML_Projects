# BigSolDB-T — Thermodynamic Inductive Bias for Temperature-Dependent Solubility

**Question:** When does thermodynamic inductive bias help temperature-dependent
solubility prediction, and when does chemical representation remain the
dominant bottleneck?

**Dataset:** BigSolDB 2.0 (Krasnov et al. 2025). After dropping rows with
missing `LogS`, 100,983 measurements across 1,445 solutes, 70 solvents,
10,855 solute-solvent pairs, temperature 243.15–425.77 K.

## Current takeaway

Van't Hoff is an excellent description of temperature dependence in BigSolDB.
A scale-conditioned Van't Hoff head learns pair-specific temperature slopes
far better than the naive implementation and improves temperature
extrapolation for known chemistry. However, the advantage disappears under
robust chemistry-shift and joint chemistry-temperature tests because the
dominant error is the chemistry-dependent intercept rather than the
temperature slope.

## Models

All share identical solute+solvent RDKit descriptors (198 + 149 = 347 dim)
and the same MLP backbone (256 → 128, dropout 0.15). Only the way temperature
enters differs.

| | T representation | Head | log S(T) |
|---|---|---|---|
| **A** direct | raw `T` appended | linear → 1 | predicted directly |
| **D** 1/T control | `[T, 1/T]` appended | linear → 1 | predicted directly |
| **B** Van't Hoff (stock) | none | linear → `(a, b)` | `a + b·(1/T_ref − 1/T)` |
| **Bc** Van't Hoff (conditioned) | none | linear → `(a, β)` | `a + β·b_scale·(1/T_ref − 1/T)` |

`T_ref = 298.15 K`. `b_scale` is the median `|b|` of per-pair OLS Van't Hoff
fits over **training rows only** for each split (≈ 1382 K on every split).
B and Bc are mathematically the same function class; Bc only rescales the
head so both outputs live on comparable numerical scales at initialization.

## Results

### 1. Van't Hoff is an excellent local description

Across 10,500 pairs with ≥ 3 distinct temperatures:

- median R² = 0.9965
- median residual RMSE = 0.010 log S
- median \|b\| = 1382 K

Not exact thermodynamics; an excellent empirical approximation over BigSolDB's
temperature ranges.

### 2. The stock Van't Hoff head has an optimization-conditioning problem

With `z = 1/T_ref − 1/T` and typical `|z| ~ 1e-4` in this dataset,

- `dL/da ~ 1`
- `dL/db ~ z ~ 1e-4`

The raw slope head receives a ~1000× smaller learning signal than the
intercept head.

### 3. Reparameterization fixes it

Bc rescales `b = β · b_scale` so `dL/dβ ~ b_scale · z ~ O(1)`. Same function
class; different conditioning. On slope recovery (per-pair effective slope vs
empirical OLS, random split, seed 42, 6,490 pairs):

| model | Pearson r | Spearman ρ | MAE (K) |
|---|---:|---:|---:|
| B (stock) | 0.19 | 0.25 | 607 |
| **Bc** (conditioned) | **0.73** | **0.69** | **386** |

### 4. Bc robustly improves T-extrapolation for known chemistry

Paired cluster bootstrap over `pair_id`, ensemble predictions across 3 model
seeds, 2000 draws:

| | test RMSE (mean over 3 seeds) |
|---|---:|
| A direct | 0.2435 |
| Bc | **0.2241** |
| **Δ = Bc − A** | **−0.019, 95 % CI [−0.022, −0.017], P(Bc better) = 1.000** |

This is the strongest positive result. When chemistry is represented in
training and the shift is along temperature, Van't Hoff bias improves
extrapolation.

### 5. Chemistry-shift improvement is not robust

The original single cold-solute partition suggested Bc might help. Five
repeated cold-solute holdouts (different held-out solute sets, 3 model seeds
each) reversed the conclusion:

- mean Δ (Bc − A) = **+0.0085**
- Bc wins **1 of 5** holdouts
- across-holdout std = 0.0093, range [−0.002, +0.022]

Cold-pair (3 repeated holdouts) is a tie: mean Δ = −0.001, Bc wins 2 of 3.

### 6. Joint (unseen chemistry + T-extrapolation) shows no Bc advantage

Joint split: hold out 145 solutes; evaluate on the upper 25 % of each eligible
held-out pair's temperature range (1,106 pairs, 2,820 test rows).

| | test RMSE (mean ± std over 3 seeds) |
|---|---|
| A direct | 0.9377 ± 0.021 |
| Bc | 0.9406 ± 0.007 |
| Δ = Bc − A (ensemble) | **+0.003, 95 % CI [−0.010, +0.016], P(Bc better) = 0.33** |

Adding T-extrapolation on top of chemistry shift does not rescue the physics
model. An important negative result: the hypothesis that Bc benefit should
scale with the T-extrapolation content of the shift is not supported.

### 7. Mechanism — intercept, not slope

Bc's residual from the empirical Van't Hoff line decomposes exactly:

`ŷ − (a* + b*·z) = da + db·z`

with `da = â − a*`, `db = b̂ − b*`. Median magnitudes on Bc test predictions:

| split | median \|da\| (intercept) | median \|db·z\| (slope) | ratio |
|---|---:|---:|---:|
| T-extrap | 0.083 | 0.064 | 1.3× |
| **cold-solute** | **0.527** | 0.038 | **13.7×** |

Bc learns the temperature-response shape reasonably well; for unseen
chemistry it places the whole curve at the wrong vertical position. Van't
Hoff constrains the slope, not the intercept. This is the mechanism behind
findings 5 and 6.

### 8. Methodological note

The notebook-07 fixed-split cold-solute bootstrap CI [−0.029, −0.006]
described uncertainty from resampling *examples within one partition*.
Across-partition variability, measured in notebook 08 by repeated group
holdouts, is larger and reverses the conclusion. For chemistry-OOD claims,
repeated group holdouts are the appropriate uncertainty quantification.

## Layout

```
data/       BigSolDBv2.0.csv, densities
notebooks/
  01_audit                     dataset integrity + counts
  02_prepare_features          RDKit descriptors -> features.npz
  03_make_splits               5 frozen splits -> splits.npz
  04_train / 05_summarize      historical single-seed workflow (kept for reference)
  06_multiseed_benchmark       A/D/B/Bc x 5 splits x 3 seeds (60 runs)
  07_vanthoff_diagnostics      per-pair fits, oracle, slope recovery, bootstrap, error decomposition
  08_distribution_shift_stress_tests  repeated cold-sol + cold-pair; joint chem+T split
scripts/
  models.py                    A / D / B / Bc definitions
  train.py                     reusable training loop
  prepare_b_scale.py           training-only per-split b_scale
  verify_parity.py             one-shot parity harness (see results/parity_report.md)
results/
  metrics.csv                          historical seed-42 A/D/B (untouched)
  parity_report.md                     Phase-1 refactor bit-parity, max |ΔRMSE| = 4.55e-08
  b_scale.json                         per-split training-only b_scale
  metrics_descriptor_multiseed.csv     notebook 06 combined 60-run table
  metrics_stage{4,5}.csv               notebook 06 A/D/B and Bc separately
  metrics_repeated_coldsol.csv         notebook 08 repeated cold-solute holdouts
  metrics_repeated_coldpair.csv        notebook 08 repeated cold-pair holdouts
  metrics_coldsol_textrap.csv          notebook 08 joint chem+T split
  bootstrap_bc_vs_a.csv                notebook 07 paired bootstrap CIs
  preds/                               per-run npz (test_idx, pair_id, T, y_true, y_pred, slope_a, slope_b)
  preds_coldsol_textrap/               per-run npz for the joint split
  fig07_*.png, fig08_*.png             analysis figures
```

## Reproducing

```
notebooks/01_audit.ipynb                             # dataset audit
notebooks/02_prepare_features.ipynb                  # -> results/features.npz
notebooks/03_make_splits.ipynb                       # -> results/splits.npz
notebooks/06_multiseed_benchmark.ipynb               # ~60 min (60 training runs)
notebooks/07_vanthoff_diagnostics.ipynb              # ~2 min (analysis only)
notebooks/08_distribution_shift_stress_tests.ipynb   # ~40 min (54 training runs)
```

Training is bit-deterministic on a fixed machine, not cross-platform. The
refactor of the historical training loop into `scripts/train.py` was verified
to `max |ΔRMSE| = 4.55e-08` against `results/metrics.csv` on all 15 seed-42
A/D/B combos (`scripts/verify_parity.py`).

## Next

1. **Notebook 09 — temperature-data sparsity.** Test whether thermodynamic
   inductive bias reduces the amount of temperature data needed per chemical
   system.
2. **Notebook 10 — GINE representation study.** Test whether a stronger
   learned molecular representation improves the chemistry-dependent
   intercept and changes the value of thermodynamic bias under chemical
   shift.
