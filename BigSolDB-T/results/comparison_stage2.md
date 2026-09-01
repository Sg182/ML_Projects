# Stage-2 comparison — A vs D vs B on 5 splits (seed 42)

| Model | Split | Test RMSE | Test R² | ΔRMSE vs random |
|---|---|---:|---:|---:|
| A direct | random | 0.201 | 0.972 |   0.000 |
| A direct | T-extrap | 0.241 | 0.960 | +0.040 |
| A direct | cold-solute | 0.871 | 0.489 | +0.670 |
| A direct | cold-pair | 0.556 | 0.799 | +0.354 |
| A direct | cold-solvent | 0.449 | 0.829 | +0.247 |
| D 1/T control | random | 0.231 | 0.963 |   0.000 |
| D 1/T control | T-extrap | 0.252 | 0.956 | +0.022 |
| D 1/T control | cold-solute | 0.893 | 0.463 | +0.662 |
| D 1/T control | cold-pair | 0.551 | 0.802 | +0.321 |
| D 1/T control | cold-solvent | 0.438 | 0.837 | +0.208 |
| B Van't Hoff | random | 0.278 | 0.947 |   0.000 |
| B Van't Hoff | T-extrap | 0.328 | 0.925 | +0.050 |
| B Van't Hoff | cold-solute | 0.893 | 0.463 | +0.615 |
| B Van't Hoff | cold-pair | 0.566 | 0.792 | +0.288 |
| B Van't Hoff | cold-solvent | 0.524 | 0.767 | +0.246 |