# Design note — GINE representation experiment

Written before running the experiment, to fix the hypothesis, the comparison and
the interpretation criteria in advance.

## Motivation

Error under chemical distribution shift is not distributed evenly between the
two Van't Hoff parameters. For held-out solutes the prediction error decomposes
as `err = da + db·z` with `|z| ≤ 1e-3`, and the intercept term dominates the
slope term by roughly an order of magnitude. The model gets the shape of the
solubility–temperature curve approximately right and places it at the wrong
absolute level.

The intercept `a` is a chemistry quantity: it is the solubility at the reference
temperature, predicted from the solute and solvent representation alone. The
temperature law constrains how solubility *changes*, not how soluble a molecule
is to begin with. So if the remaining bottleneck is `a`, it is a representation
problem rather than a thermodynamics problem.

RDKit descriptors are fixed, hand-designed functions of a molecule. A learned
graph encoder could in principle capture solute–solvent structure that those
descriptors miss. Whether it actually does is an empirical question.

## Hypothesis

A learned molecular graph representation (GINE) reduces the intercept error
`|a_pred − a_true|` for unseen solutes, relative to RDKit descriptors, and
thereby improves cold-solute generalisation.

A secondary question is whether representation and thermodynamic bias interact:
does a better encoder make the Van't Hoff head more useful, or do the two
address independent parts of the error?

## Models compared

A 2×2 crossing representation with prediction head, so that the two factors can
be separated:

| | direct head | Van't Hoff head |
|---|---|---|
| RDKit descriptor MLP | A_MLP | Bc_MLP |
| GINE encoder | A_GINE | Bc_GINE |

Solute and solvent are encoded separately — the dataset contains far more
solutes than solvents, and sharing encoder weights would impose an unwarranted
prior. Within a column the head is identical; within a row the representation
is identical. Backbone width, dropout, optimiser, learning rate, batch size,
early-stopping rule and seeds are held fixed across all four cells, so the only
varying factors are the two of interest.

GINE is deliberately kept small (3 layers, hidden 128) and untuned. The purpose
is a controlled representation comparison, not an architecture search.

## Evaluation splits

- **random** — parity check (see below)
- **T-extrapolation** — unseen temperatures, known chemistry
- **cold-pair** — unseen pairings of molecules seen elsewhere in training
- **cold-solute** — unseen molecules; the split the hypothesis is about
- **joint** — unseen solutes evaluated on their unseen high temperatures

Cold-solvent is left out: with only 70 solvents in the dataset, a held-out set
is too small to support a conclusion.

### Parity requirement

Before any out-of-distribution comparison is interpreted, `A_GINE` must be
competitive with `A_MLP` on the random split. It need not win. But if the graph
encoder is substantially worse in distribution, then it is not a fair
representation and any OOD difference would confound representation quality
with implementation quality.

## Diagnostics

RMSE alone cannot answer the question, because the hypothesis is about a
specific component of the error. Alongside the split-level metrics:

- **intercept error** — median `|a_pred − a_true|` per held-out pair, comparing
  `Bc_MLP` against `Bc_GINE`. This is the quantity the hypothesis predicts will
  improve.
- **slope recovery** — Pearson correlation between predicted and empirical
  per-pair `b`, to check whether the two parameters behave differently.
- **interaction** — `Δ_phys = RMSE(direct) − RMSE(Van't Hoff)` computed for each
  representation, then compared between them.

## Interpretation criteria

- If GINE reduces cold-solute RMSE *and* intercept error, the chemistry
  bottleneck is representational and a stronger encoder is the productive
  direction.
- If GINE improves both heads by a similar amount, representation and
  thermodynamic bias solve largely independent problems, and the physics result
  stands on its own.
- If GINE does not reduce cold-solute or intercept error despite matching the
  descriptor model on the random split, this would indicate that replacing fixed
  descriptors with this graph encoder alone does not resolve the
  chemistry-generalisation bottleneck.

The third case is informative rather than a failure: it would localise the
limitation away from both the temperature law and the encoder family, and toward
data coverage or the intrinsic difficulty of predicting absolute solubility for
unfamiliar chemistry.

## Anticipated limitations

- Cold-solute holds out roughly 145 solutes, and a single partition has already
  proved capable of producing an overconfident result on this axis. Any
  cold-solute conclusion should be checked against repeated group holdouts
  before it is trusted.
- GINE has more capacity than the descriptor MLP and is not tuned. A negative
  result therefore constrains this configuration on this dataset, not graph
  neural networks in general.
- Running the full 2×2 across five splits and three seeds is expensive; the
  random-split parity check is run first so that the remaining cost is only
  incurred if the comparison is meaningful.
