# Dataset

This repository uses **BigSolDB 2.0**, created by Lev Krasnov, Dmitry Malikov,
Marina Kiseleva, Sergei Tatarin, Sergey Sosnin and Stanislav Bezzubov.

**BigSolDB 2.0 is third-party research data and was not created by the author of
this repository.** It is included here for convenience and reproducibility, with
the original filenames unchanged.

The original dataset, metadata and provenance are available from Zenodo:
[`10.5281/zenodo.15094979`](https://doi.org/10.5281/zenodo.15094979)

## Citation

If you use this repository or the included dataset, please cite:

> Krasnov, L., Malikov, D., Kiseleva, M., Tatarin, S., Sosnin, S., and
> Bezzubov, S. *BigSolDB 2.0, dataset of solubility values for organic compounds
> in different solvents at various temperatures.* Scientific Data **12**, 1236
> (2025). DOI: [`10.1038/s41597-025-05559-8`](https://doi.org/10.1038/s41597-025-05559-8)

## Included files

| File | Description |
|---|---|
| `BigSolDBv2.0.csv` | main solubility dataset — used throughout this project |
| `BigSolDBv2.0_densities.csv` | solvent-density data from the source release — included for completeness; **not read by any notebook or script here** |

The official dataset record states that BigSolDB 2.0 contains 103,944
experimentally measured solubility values for 1,448 compounds in 213 solvents.

## Subset used in this project

`notebooks/01_audit.ipynb` reproduces the audit. The reported results are
computed after dropping rows with a missing `LogS(mol/L)` value:

| quantity | value |
|---|---|
| rows used | 100,983 |
| unique solutes | 1,445 |
| unique solvents | 70 |
| unique solute–solvent pairs | 10,855 |
| temperature range | 243.15 – 425.77 K |
| target | `LogS(mol/L)`, base-10 |

The difference from the headline record counts (103,944 / 1,448 / 213) is
entirely due to that `LogS` filter; the CSV itself is unmodified.

## Licensing

The MIT License in this repository applies to the original code in this project.
Third-party data, including BigSolDB 2.0, remains subject to the terms and
attribution requirements of its original source.
