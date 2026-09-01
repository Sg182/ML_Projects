# Run log — Stage 1 MVP first results

**Date:** 2026-09-01
**Seed:** 42 (single seed for debugging; multi-seed reserved for reported comparison)
**Device:** CPU
**Data:** BigSolDB 2.0 primary CSV (100,983 rows after dropping missing LogS)

## Aggregate results

Table — RMSE / MAE / R² (test set) per model and split, seed 42.

| Model | Split | Train RMSE | Val RMSE | **Test RMSE** | Test MAE | Test R² | Time |
|---|---|---|---|---|---|---|---|
| **A** direct (T)          | random   | 0.1741 | 0.1885 | **0.2013** | 0.1305 | 0.9721 | 148 s |
| **D** 1/T control ([T,1/T]) | random   | 0.2026 | 0.2142 | **0.2305** | 0.1534 | 0.9634 |  92 s |
| **B** Van't Hoff physics   | random   | 0.2600 | 0.2695 | **0.2781** | 0.1939 | 0.9468 |  63 s |
| **A** direct (T)          | textrap  | 0.1862 | 0.1974 | **0.2409** | 0.1732 | 0.9596 | 112 s |
| **D** 1/T control ([T,1/T]) | textrap  | 0.1960 | 0.2094 | **0.2524** | 0.1824 | 0.9556 |  90 s |
| **B** Van't Hoff physics   | textrap  | 0.2459 | 0.2604 | **0.3281** | 0.2455 | 0.9251 |  59 s |

**Ranking on both splits: A (best) < D < B (worst).**

## Interpretation

**Random split is very easy.** Model A reaches test RMSE 0.20, already below the reported cross-lab aleatoric floor of 0.39 logS. This confirms the "matrix completion" intuition: random-split test contains many measurements where the model has already seen the same (solute, solvent) pair at nearby T. The interpolation task is trivial for a moderately-capable MLP with T as a feature. **This is exactly why random-split RMSE is not the meaningful metric** — a paper thesis point.

**T-extrapolation hurts all three models, but at nearly the same relative rate.**
- A: 0.201 → 0.241, +20 % relative
- D: 0.231 → 0.252, +9.5 % relative
- B: 0.278 → 0.328, +18 % relative

The physics-informed model B does NOT gain a *relative* advantage on T-extrapolation over the direct model A. It degrades similarly and is worse in absolute terms on both splits.

**Model D (adding 1/T as a feature) is intermediate on random split and no better than A on T-extrap.** So the "physics benefit" is not merely a 1/T representation effect that a direct model could recover for free.

**Sanity checks that pass:**
- Model D degrades less from random → textrap than Model A (9.5 % vs 20 %). Some hint that 1/T as a feature helps with T-extrapolation, if only modestly.
- Model B's degradation is comparable to Model A's, not the *smaller* degradation the paper hypothesis would predict.
- All models trained smoothly, no exploding losses, no NaNs.

## What this null result means for the thesis (so far)

The user's stated hypothesis was that:
1. Direct ML performs similarly or better under random interpolation. **✓ Confirmed.**
2. Physics-informed ML becomes better under T-extrapolation. **✗ NOT confirmed in this setup.**

Before treating (2) as falsified, consider:

- **T-extrap may not be hard enough with the current per-pair rule.** The split holds out the upper 25 % of T within each pair; many held-out points are still within the *global* training T range. A more aggressive rule (e.g., upper 15 %, or holding out the top 30 K of each pair regardless of proportion) may reveal a physics advantage where the current rule does not.
- **The physics-informed model B is capacity-constrained.** It predicts just 2 numbers per (solute, solvent), then composes. This is *by design* a strong prior; whether it is *too strong* depends on how well the true T-dependence follows the assumed Van't Hoff form over 243–425 K.
- **The next-scheduled generalization regimes (cold-solute, cold-solvent, cold-pair) may show the physics benefit that T-extrapolation does not.** Under cold-solute / cold-pair, the direct model cannot memorize pair-specific T-curves; the physics-informed model is on more equal footing.
- **Two-parameter Van't Hoff assumes T-independent effective enthalpy over the pair's T range.** A 3-parameter Kirchhoff extension (deferred, Model C) allows curvature. If curvature explains why B underperforms, Model C could recover it.

**Do not draw the thesis conclusion yet.** The evidence so far says: on random and T-extrapolation splits alone, the Van't Hoff physics hurts more than it helps. That is a legitimate observation that needs to be checked against harder generalization regimes.

## Files produced

- `results/audit_report.md` — dataset audit
- `results/features.npz` — cached RDKit descriptors (198 solute + 149 solvent = 347 dim)
- `results/splits.npz` — random + T-extrapolation splits
- `results/metrics.csv` — per-run metrics (append-safe)
- `results/run_log.md` — this file

## Reproduction

Run the five notebooks in order from `notebooks/`:

```
notebooks/01_audit.ipynb
notebooks/02_prepare_features.ipynb
notebooks/03_make_splits.ipynb
notebooks/04_train.ipynb          # edit COMBOS / SEED at the top if needed
notebooks/05_summarize.ipynb
```

The training notebook imports model classes from `scripts/models.py`.

---

# Stage 2 — cold-group splits (added)

**Splits added, all group-aware:**
- Cold-solute (80/10/10 at solute level; 1,156 / 144 / 145 solutes)
- Cold-pair (80/10/10 at pair level; 8,684 / 1,086 / 1,085 pairs)
- Cold-solvent (80/10/10 at solvent level; 56 / 7 / 7 solvents)

**Leakage audit** performed and all disjointness contracts satisfied (see `03_make_splits.py` output above; every "contract satisfied" line was printed).

**Caveat on cold-solvent:** the LogS-cleaned dataset has only 70 unique solvents (many rare solvents live in the 2,961 dropped rows with missing LogS). 7 held-out solvents is a small sample; results will have higher variance than the other splits. High priority for a multi-seed rerun at some point.

## Full comparison table (all 15 seed-42 runs)

| Model | Split | Test RMSE | Test R² | ΔRMSE vs random |
|---|---|---:|---:|---:|
| A direct       | random       | 0.201 | 0.972 |   0.000 |
| A direct       | T-extrap     | 0.241 | 0.960 | +0.040 |
| A direct       | cold-solute  | 0.871 | 0.489 | +0.670 |
| A direct       | cold-pair    | 0.556 | 0.799 | +0.354 |
| A direct       | cold-solvent | 0.449 | 0.829 | +0.247 |
| D 1/T control  | random       | 0.231 | 0.963 |   0.000 |
| D 1/T control  | T-extrap     | 0.252 | 0.956 | +0.022 |
| D 1/T control  | cold-solute  | 0.893 | 0.463 | +0.662 |
| D 1/T control  | cold-pair    | 0.551 | 0.802 | +0.321 |
| D 1/T control  | cold-solvent | 0.438 | 0.837 | +0.208 |
| B Van't Hoff   | random       | 0.278 | 0.947 |   0.000 |
| B Van't Hoff   | T-extrap     | 0.328 | 0.925 | +0.050 |
| B Van't Hoff   | cold-solute  | 0.893 | 0.463 | +0.615 |
| B Van't Hoff   | cold-pair    | 0.566 | 0.792 | +0.288 |
| B Van't Hoff   | cold-solvent | 0.524 | 0.767 | +0.246 |

## Rankings

### By absolute test RMSE (best → worst per split)

| Split | Best | Middle | Worst |
|---|---|---|---|
| random       | A (0.201) | D (0.231) | B (0.278) |
| T-extrap     | A (0.241) | D (0.252) | B (0.328) |
| cold-solute  | A (0.871) | D (0.893) | B (0.893) |
| cold-pair    | D (0.551) | A (0.556) | B (0.566) |
| cold-solvent | D (0.438) | A (0.449) | B (0.524) |

### By relative degradation ΔRMSE (smallest = most robust)

| Split | Most robust | Middle | Least robust |
|---|---|---|---|
| T-extrap     | D (+0.022) | A (+0.040) | B (+0.050) |
| **cold-solute**  | **B (+0.615)** | D (+0.662) | A (+0.670) |
| **cold-pair**    | **B (+0.288)** | D (+0.321) | A (+0.354) |
| cold-solvent | D (+0.208) | B (+0.246) | A (+0.247) |

## Interpretation (conservative)

**1. Chemistry identity is a much harder generalization axis than temperature.**
Order of difficulty (test RMSE ranges over A/D/B):
- Random 0.20–0.28 (easy)
- T-extrap 0.24–0.33 (modest)
- Cold-solvent 0.44–0.52 (moderate; high variance from 7 test solvents)
- Cold-pair 0.55–0.57 (moderate)
- Cold-solute 0.87–0.89 (hard)

Cold-solute test R² collapses to ~0.46 for all three models. Random-split R² of ~0.97 does not reflect out-of-distribution performance on this axis at all.

**2. On chemistry-identity shifts (cold-solute, cold-pair), Model B degrades the least.**
- Cold-solute: A +0.670, D +0.662, **B +0.615** — B is 8% less degradation than A.
- Cold-pair:   A +0.354, D +0.321, **B +0.288** — B is 19% less degradation than A.
This is the first sign of a physics-consistent robustness advantage. It matches the user-specified hypothesis that thermodynamic inductive bias could show up as a smaller ΔRMSE on hard splits even if not as a better absolute number. It is present on the two hardest chemistry-identity axes.

**3. Model B remains worst or tied-worst in absolute test RMSE on every split.**
The absolute Van't Hoff-informed number is never below the direct baseline. Robustness only shows up in relative degradation, not absolute performance.

**4. The result is only preliminary.**
- Single seed.
- Val sets are group-disjoint from train (as required) but may sample harder/easier subsets than test by chance, especially on cold-solvent (7 val solvents / 7 test solvents).
- Cold-solvent has high sample-size variance and should not carry weight until multi-seeded.
- Early stopping used val RMSE; for group-cold splits val does not reflect the same difficulty as test perfectly.

**5. What this means for the thesis.**
The user's hypothesis that "physics-informed ML improves robustness under chemical distribution shift" is *partially and conservatively* supported by these results on the two chemistry-identity splits (cold-solute, cold-pair). It is not supported on T-extrapolation. It is inconclusive on cold-solvent because of the small held-out solvent count.

The user's separate framing — that a paper claim could be "same or worse absolute RMSE but smaller degradation on hard splits" — is exactly the pattern we see for B vs A on cold-solute and cold-pair. Whether the effect survives multi-seed replication is the next quantitative question.

## Correction on earlier wording

Stage 1 characterized random-split as "trivial" because test RMSE (~0.20) fell below the cross-lab replicate RMSE of 0.39 log units reported by Krasnov et al. That comparison is not clean without demonstrating the two measurements use directly comparable subsets and error definitions. The safer statement is: **random splitting appears very easy because of chemical/pair overlap between train and test**, and BigSolDB 2.0's reported 0.39 aleatoric figure is not directly comparable to our test-set RMSE without matched-subset validation.

## Do NOT modify Model B in response to this result

Per the user directive: no Kirchhoff correction, no architecture changes, no stricter T-extrap during this stage. If the next scientific step is warranted after review, it would be either:
- multi-seed replication of the current 15-run table to quantify the ΔRMSE gap uncertainty for B vs A on cold-solute and cold-pair, or
- Kirchhoff Model C to test whether Van't Hoff curvature limitations explain B's worse absolute performance.

**Stopping here as instructed. Awaiting review.**

