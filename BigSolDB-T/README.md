# BigSolDB-T — Thermodynamic Inductive Bias for Solubility Extrapolation

**Question:** does thermodynamic inductive bias improve solubility prediction
under temperature and chemical distribution shift?

**Scope:** a narrow methodology study on BigSolDB 2.0 (100,983 measurements
after dropping missing `LogS`; 1,448 solutes, 70 solvents, 10,855 pairs)
comparing matched-encoder models. Not an architecture race.

Status: **Stages 1–6 complete.** Stage 7 (GINE) designed, not implemented.

---

## Models

All share one MLP backbone (256, 128), identical RDKit descriptors
(198 solute + 149 solvent = 347), and identical training. Only temperature
handling differs.

| | T representation | Head | logS(T) |
|---|---|---|---|
| **A** direct | raw `T` appended | → 1 | predicted directly |
| **D** 1/T control | `[T, 1/T]` appended | → 1 | predicted directly |
| **B** Van't Hoff (stock) | none | → (a, b) | `a + b·(1/T_ref − 1/T)` |
| **Bc** Van't Hoff (conditioned) | none | → (a, β) | `a + β·b_scale·(1/T_ref − 1/T)` |

`b_scale` = median \|b\| of per-pair OLS fits over **training pairs only**
(1381.8 K). Bc is a *reparameterization* of B — identical function class.
Sign conventions and the ΔH derivation: `docs/vanthoff_conventions.md`.

## Headline results (test RMSE, 3 seeds: 42/123/456)

| Split | A | D | B (stock) | **Bc** |
|---|---:|---:|---:|---:|
| random | **0.2023** | 0.2203 | 0.2584 | 0.2033 |
| T-extrapolation | 0.2376 | 0.2401 | 0.3073 | **0.2204** |
| cold-solute | 0.8997 | 0.9003 | 0.8870 | **0.8826** |
| cold-pair | 0.5540 | 0.5549 | 0.5599 | **0.5483** |
| cold-solvent\* | 0.4530 | 0.4624 | 0.4614 | **0.4413** |

\*exploratory — only 7 held-out solvents; no claim drawn.

Paired cluster bootstrap, Bc − A: **T-extrap [−0.0197, −0.0147], P = 1.00**;
cold-solute [−0.0406, +0.0087], P = 0.89; cold-pair [−0.0214, +0.0081], P = 0.77.

## Observations

1. **Random splits are misleading.** Test R² ≈ 0.97 on random vs ≈ 0.46 on
   cold-solute. Chemistry identity is a far harder axis than temperature.

2. **The conditioned physics model wins on temperature.** Bc beats A on
   T-extrapolation by 0.017 log units, consistent across seeds and the only
   comparison whose bootstrap CI excludes zero. Mechanism confirmed: Bc recovers
   per-pair Van't Hoff slopes at r = 0.49 vs A's 0.31.

3. **It does not win on chemistry.** Bc leads nominally on all three cold splits
   and beats D consistently, but every Bc−A interval straddles zero and one
   cold-solute seed reverses sign. Treat as a tie with a slight lean.

4. **The bottleneck is the intercept, not the thermodynamics.** Errors decompose
   exactly as `err = da + db·z` with `|z| ≤ 1e-3`. Under cold-solute the
   intercept error is **12×** the slope's entire contribution. Predicted curves
   run parallel to observations but sit vertically offset — right temperature
   response, wrong absolute solubility.

5. **Chemical similarity does not predict who fails.** Intercept error vs
   nearest-training-neighbour Tanimoto: ρ = −0.03 (no trend across quartiles).
   Training coverage does track it (ρ = −0.18, monotone). The limit is what
   RDKit descriptors encode, not distance from the training set.

6. **Two corrections recorded** (both are results in their own right):
   - The "degradation relative to random split" metric is **not** robustness
     evidence. It equals `absolute difference − random-split gap`, so it
     mechanically rewards a weak in-distribution baseline.
   - Stock model B did **not** fail for lack of expressive capacity — its
     trained slopes reached a median 1308 K against an empirical 1381 K. It
     failed from optimization conditioning: two head outputs differing ~1000× in
     natural scale, with the slope's gradient attenuated by `z ~ 1e-3`.

7. **Constant-ΔH is a good description.** Per-pair Van't Hoff fits give median
   R² = 0.996, residual 0.010 log units. Held-out Van't Hoff oracle 0.099 vs
   T-blind 0.429. Kirchhoff curvature is therefore *not* the priority.
   (The earlier 0.076 figure is an **in-sample fit error**, not a
   generalization floor.)

## Layout

```
data/        BigSolDBv2.0.csv (unmodified), densities
notebooks/   01_audit → 02_prepare_features → 03_make_splits → 04_train → 05_summarize
scripts/     models.py (A/D/B/Bc)
             run_stage4.py      3-seed A/D/B, all 5 splits
             run_stage5.py      3-seed Bc, all 5 splits
             analyze_stage4.py  seed stability + bootstrap
             analyze_stage6.py  A/D/Bc benchmark + bootstrap
             stage5_slopes.py   empirical slopes, units, oracles
             stage5_scale_check.py / stage5_sanity.py / stage5_curves.py
             stage6_similarity.py (RDKit env) / stage6_diagnose.py
results/     stage{4,5,6}_summary.md  ← start here
             metrics_stage{4,5}.csv, preds/, figs/, model_Bc_*.pt
docs/        vanthoff_conventions.md   frozen formulation + manuscript notes
             stage7_gine_design.md     next experiment (not implemented)
```

## Reproducing

```
python3 scripts/run_stage4.py          # A/D/B  × 5 splits × 3 seeds  (~22 min)
python3 scripts/run_stage5.py          # Bc     × 5 splits × 3 seeds  (~10 min)
python3 scripts/analyze_stage4.py
python3 scripts/stage5_slopes.py && python3 scripts/stage5_sanity.py
/path/to/envs/ml/bin/python scripts/stage6_similarity.py   # needs RDKit
python3 scripts/analyze_stage6.py && python3 scripts/stage6_diagnose.py
```

Notes: import `sklearn` before `torch` (libomp clash on macOS). Training is
bit-deterministic on a fixed machine but *not* across platforms — Stage-4 seed-42
numbers differ from the original Linux `metrics.csv` in the 3rd–4th decimal, so
all reported comparisons were re-run on one machine.

## Next steps

1. **Stage 7 — GINE, as a test of the representation bottleneck.** Run the 2×2
   (descriptor vs graph encoder) × (direct vs physics head) so the *interaction*
   is the measured quantity. Gate on a descriptor-parity check (A_GINE ≈ A_MLP
   on random) before interpreting anything OOD. Design and pre-registered
   predictions in `docs/stage7_gine_design.md`. **The novelty stays "does
   thermodynamic bias help generalization?" — not "we used a GNN".**
2. **Fix cold-solute's resolving power.** Its bootstrap CI is ±0.04 wide against
   145 held-out solutes; any GINE gain smaller than that is unresolvable.
   Repeated group holdouts before treating it as decisive.
3. **Redesign the cold-solvent evaluation** (7 solvents is too few) if that split
   is to carry any claim.

Deferred, in priority order: repeated-holdout evaluation → Kirchhoff Model C
(low priority, see observation 7) → GAT → uncertainty quantification →
hyperparameter sweeps.
