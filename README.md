# Machine learning projects

Independent machine learning projects on real datasets, each with its own directory, notebooks,
results, and a README describing the problem, method, and findings.

**[ERCOTLoadForecasting](ERCOTLoadForecasting)** — day-ahead electricity demand forecasting for
the Texas grid under a fixed 10:00 AM issuance protocol that uses only information available at
that cutoff. Gradient boosting reaches a test MAE of 3,073 MW against a protocol-valid naive
baseline's 3,630 MW over the 8,761-hour 2025 test set. Includes a live feature pipeline, a FastAPI
day-ahead endpoint, a Dockerfile, and a test suite run in GitHub Actions.

**[BigSolDB-T](BigSolDB-T)** — whether encoding the van't Hoff relation into a neural network
improves temperature extrapolation for molecular solubility. A matched pair of models differing
only in how temperature enters is evaluated across held-out regimes spanning temperature
extrapolation, unseen solutes, unseen solute–solvent pairs, and the two combined.

**[ESOL](ESOL)** and `NN_scratch.ipynb` are exploratory: molecular representations and
scaffold-aware splitting on AqSolDB, and a neural network implemented from scratch.

The emphasis throughout is on evaluation that survives contact with reality — protocols restricted
to information available at prediction time, baselines chosen to be hard rather than flattering,
and negative results reported as negative results.
