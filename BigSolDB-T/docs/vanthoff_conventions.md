# Frozen Van't Hoff formulation and sign conventions

Reference for the sign conventions used throughout the code and the figures.
Every equation below uses one convention.

Implementation: `VantHoffScaledModel` (kind `Bc`) in `scripts/models.py`. The
configuration is held fixed across all reported experiments: hidden layers (256, 128), dropout 0.15,
RDKit descriptor set (198 solute + 149 solvent = 347), AdamW (lr 1e-3,
weight_decay 1e-5), batch 512, max 100 epochs, early stopping patience 10 on
validation RMSE, and the split files in `results/splits.npz`.

---

## 1. Target

`y = log10 S`, S in mol/L. Base-10 verified: max |LogS - log10(Solubility)|
= 4.1e-13 over all 100,983 rows.

## 2. Model equation (canonical form)

```
x_T   = b_scale * (1/T_ref - 1/T)          T_ref = 298.15 K
y_hat = a + beta * x_T
```

`a` and `beta` are the two outputs of the network head. `b_scale` is a fixed
constant, not learned.

## 3. b_scale

```
b_scale = median | b_m | over the pairs appearing in the TRAINING rows
        = 1381.8 K   (random and textrap splits; refitted per split)
```

Derived from training data only — validation and test pairs are never
consulted. Stored as a non-trainable buffer in the checkpoint.

## 4. Recovering the physical slope

```
b_m = beta * b_scale
```

`b_m` is the slope in **model convention**, defined by

```
y = a + b_m * (1/T_ref - 1/T)
```

Because `(1/T_ref - 1/T)` increases with T, **`b_m > 0` means solubility rises
with temperature** (endothermic dissolution). This is the sign used in every
table and figure in this project.

## 5. Relation to the conventional Van't Hoff slope

Written against `1/T` directly — the form used when fitting per-pair OLS to the
raw data ("data convention"):

```
y = a_d + b_d * (1/T)
```

Expanding the model equation gives the coefficient of `(1/T)` as `-b_m`, so

```
b_d = -b_m           and        a_d = a + b_m / T_ref
```

`b_d` is negative for a typical endothermic pair. **This is the only sign flip
in the project**, and it exists solely because the two forms measure the slope
against opposite-signed abscissae.

## 6. Apparent dissolution enthalpy

```
van't Hoff     ln S = -dH_app / (R T) + const
base 10        y = -dH_app / (ln10 * R * T) + const
               b_d = -dH_app / (ln10 * R)

dH_app = -ln10 * R * b_d = + ln10 * R * b_m          ln10 * R = 19.1448 J/mol/K
```

So `b_m > 0`  <=>  `b_d < 0`  <=>  `dH_app > 0` (endothermic). Median observed
`dH_app` = +26.4 kJ/mol; 99% of pairs endothermic.

**Terminology.** `dH_app` is an *apparent* dissolution enthalpy, never
`dH_sol`. It assumes molar concentration in place of mole fraction, a
temperature-independent enthalpy across the pair's range, and neglects solvent
volume expansion. Report it as a plausibility check, not a measurement.

## 7. Summary table

| symbol | meaning | typical value | sign for endothermic |
|---|---|---:|---|
| `beta` | raw head output | O(1) | positive |
| `b_scale` | fixed constant, training-derived | 1381.8 K | positive |
| `b_m = beta * b_scale` | slope vs `(1/T_ref - 1/T)` — **project standard** | ~1381 K | **positive** |
| `b_d = -b_m` | slope vs `(1/T)` — raw OLS fits | ~-1381 K | negative |
| `a` | intercept; equals `y` at `T = T_ref` | ~-1.03 | — |
| `dH_app = ln10*R*b_m` | apparent dissolution enthalpy | +26.4 kJ/mol | positive |

## 8. Notes on interpretation

**On the unconditioned model.** Model B does not fail for lack of expressive
capacity. Its trained slopes reach a median of 1308 K against an empirical
median of 1381 K, and its temperature term carries a standard deviation of
0.240 log units against the 0.229 it needs to explain. The failure is one of
optimisation conditioning: the head must emit two quantities whose natural
scales differ by a factor of about 1000, while the slope's gradient is
attenuated by `(1/T_ref − 1/T) ~ 1e-3`. The scaled inverse-temperature
parameterisation preserves the functional form exactly and relocates a constant.

Bc is therefore a better-conditioned parameterisation of the same physical
prior, not a different physical model.

**On the scope of the benefit.** The thermodynamic bias improves temperature
extrapolation (bootstrap CI [−0.0197, −0.0147] against the direct baseline) and
preserves interpolation accuracy. It does not measurably improve generalisation
to unseen chemistry: all cold-split intervals include zero. The reason is
quantitative — under cold-solute the intercept error is roughly 14 times the
slope's contribution, so the temperature model can only address a small part of
the error.

**On evaluation.** Degradation relative to the random split is not evidence of
robustness. It equals the absolute difference minus the random-split gap, so it
rewards a model with a worse in-distribution baseline. Absolute error on the
shifted test set is the meaningful quantity.
