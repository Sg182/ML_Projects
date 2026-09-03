# Parity report — scripts/train.py vs the historical notebook loop

Platform: macOS-26.6.2-arm64-arm-64bit, python 3.11.5, torch 2.7.1, 6 threads.

## Refactor parity (same machine)

Against `results/metrics_stage4.csv` — the same 15 combinations run on this machine by the pre-refactor loop. This is the comparison that isolates the refactor.

- combinations: **15**
- **max |delta test RMSE| = 0.000e+00**

**Bit-exact.** `train.py` reproduces the pre-refactor loop with zero deviation.

## Cross-platform comparison

Against `results/metrics.csv`, the original seed-42 A/D/B runs, which were produced on different hardware (its audit log records a Linux path).

- comparisons: **45** (train/val/test RMSE for 15 combinations)
- **max |delta RMSE| = 6.647e-02**  (B/coldsol, train)
- median |delta RMSE| = 7.012e-03

| model | split | max \|delta\| over phases |
|---|---|---:|
| A | coldpair | 4.80e-03 |
| A | coldsol | 5.53e-02 |
| A | coldsolv | 9.27e-03 |
| A | random | 1.01e-02 |
| A | textrap | 4.82e-03 |
| B | coldpair | 1.29e-02 |
| B | coldsol | 6.65e-02 |
| B | coldsolv | 9.30e-03 |
| B | random | 5.22e-02 |
| B | textrap | 1.12e-02 |
| D | coldpair | 4.66e-02 |
| D | coldsol | 4.55e-02 |
| D | coldsolv | 4.45e-02 |
| D | random | 5.32e-03 |
| D | textrap | 5.60e-03 |

These deviations are far larger than float noise (~1e-4) because early stopping is a **discrete** choice: a tiny difference in float reduction order changes which epoch wins on validation, and therefore which checkpoint is kept. All three models are affected, and the magnitudes are largest on the cold splits, where validation curves are flattest and epoch selection is close to tied.

Cross-platform numbers are therefore not a refactor check. Reproduce on one machine and compare within it.
