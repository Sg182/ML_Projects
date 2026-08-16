# ESOL Molecular Solubility Prediction with Machine Learning

This project explores supervised machine-learning approaches for predicting the aqueous solubility of molecules using the **ESOL (Delaney) molecular solubility dataset**.

The goal is to build an end-to-end regression workflow starting from exploratory data analysis and simple models, followed by nonlinear ensemble methods and hyperparameter tuning with cross-validation.

## Problem

For each molecule, the objective is to predict the experimentally measured log solubility:

[
y = \log_{10}(S),
]

where (S) is the aqueous solubility in mol/L.

This is therefore a **supervised regression problem**.

The dataset contains **1,128 molecules** and includes molecular descriptors, SMILES representations, measured solubilities, and the original ESOL model prediction.

## Features

The initial models use six numerical molecular descriptors:

* Molecular Weight
* Polar Surface Area
* Number of Rings
* Number of Rotatable Bonds
* Number of H-Bond Donors
* Minimum Degree

The target variable is:

`measured log solubility in mols per litre`

The dataset also contains molecular structures represented as **SMILES strings**, which are reserved for future extensions using RDKit descriptors and molecular fingerprints.

## Exploratory Data Analysis

The dataset was first inspected for:

* Data types and feature ranges
* Missing values
* Duplicate observations
* Target distribution
* Potential target outliers
* Molecular characteristics of poorly soluble compounds

No missing values were present in the dataset.

### Low-solubility outliers

Using the (1.5\times IQR) criterion, **17 molecules** were identified as target outliers. All occurred on the extremely low-solubility side of the distribution.

These compounds had:

| Property                  |   Mean |
| ------------------------- | -----: |
| Molecular Weight          | 339.81 |
| Number of Rings           |   3.06 |
| Number of Rotatable Bonds |   3.00 |
| Number of H-Bond Donors   |   0.06 |
| Polar Surface Area        |   6.31 |
| Measured logS             |  -8.91 |

The combination of relatively high molecular weight, multiple rings, and particularly low polarity is consistent with very poor aqueous solubility.

These observations were **retained** rather than removed because extreme solubility values can represent physically meaningful molecules rather than erroneous data.

## Model Evaluation Strategy

The data were separated into training and held-out test sets.

Hyperparameters were selected using **5-fold cross-validation on the training data**. 

 

## Models

Several regression approaches were compared.

### Polynomial Ridge Regression

Polynomial features were generated to introduce nonlinear terms and feature interactions such as

[
MW^2,
\qquad
PSA^2,
\qquad
MW\times PSA.
]

The resulting features were standardized before applying Ridge regularization.

Grid-search parameters included:

* Polynomial degree
* Ridge regularization strength (\alpha)

Best parameters:

```text
Polynomial degree = 3
Ridge alpha = 1.0
```

### Random Forest

Random Forest regression was used to model nonlinear relationships and feature interactions without explicitly constructing polynomial features.

Best parameters:

```text
max_depth = None
max_features = sqrt
min_samples_leaf = 1
n_estimators = 200
```

### Gradient Boosting

Gradient Boosting sequentially constructs regression trees, with each new tree attempting to reduce errors made by the existing ensemble.

Hyperparameters explored included:

* Number of estimators
* Learning rate
* Maximum tree depth
* Minimum samples per leaf
* Subsampling fraction

### XGBoost

XGBoost was evaluated as an optimized gradient-boosted-tree implementation.

Best parameters:

```text
colsample_bytree = 1.0
learning_rate = 0.1
max_depth = 3
n_estimators = 500
subsample = 0.8
```

The complete XGBoost grid search and evaluation required approximately **9.6 seconds** on the development machine.

## Results

| Model             | Best CV RMSE |  Test RMSE |    Test R² |
| ----------------- | -----------: | ---------: | ---------: |
| Polynomial Ridge  |       0.9615 |     1.0565 |     0.7638 |
| Random Forest     |       0.8527 |     0.8748 |     0.8381 |
| Gradient Boosting |            — |     0.8759 |     0.8377 |
| XGBoost           |   **0.8180** | **0.8744** | **0.8383** |

The tree-based ensemble models substantially outperformed Polynomial Ridge regression.

XGBoost obtained the best cross-validation RMSE, while Random Forest, Gradient Boosting, and XGBoost produced nearly identical performance on the held-out test set.

Therefore, the test results should be interpreted as

[
\text{Random Forest}
\approx
\text{Gradient Boosting}
\approx
\text{XGBoost},
]

rather than claiming that one model decisively outperforms the others.

The results suggest that **nonlinear relationships and interactions between molecular descriptors are important for predicting aqueous solubility**.

## Random Forest Feature Importance

The Random Forest model produced the following impurity-based feature importances:

| Feature                   | Importance |
| ------------------------- | ---------: |
| Molecular Weight          |     0.4496 |
| Polar Surface Area        |     0.1956 |
| Number of Rings           |     0.1756 |
| Number of Rotatable Bonds |     0.1111 |
| Number of H-Bond Donors   |     0.0561 |
| Minimum Degree            |     0.0119 |

Molecular Weight was the dominant feature, followed by Polar Surface Area and Number of Rings.

Minimum Degree contributed very little predictive information.

These values represent model-based feature importance and should not be interpreted directly as physical causal importance.

## Main Findings

1. Tree-based ensemble methods significantly outperform polynomial Ridge regression for the six molecular descriptors used here.

2. Random Forest, Gradient Boosting, and XGBoost achieve similar generalization performance on the held-out test set.

3. Molecular Weight is the most influential descriptor in the Random Forest model, followed by Polar Surface Area and Number of Rings.

4. Extremely insoluble molecules tend to combine relatively large molecular size with low polarity.

5. The strong performance of tree ensembles suggests that the relationship between molecular properties and aqueous solubility contains important nonlinearities and feature interactions.

## Project Structure

```text
ESOL/
│
├── README.md
├── delaney-processed.csv
├── Data_processing.ipynb
└── Model_training.ipynb
```

`Data_processing.ipynb` contains data inspection and exploratory analysis.

`Model_training.ipynb` contains model training, cross-validation, hyperparameter optimization, and final evaluation.

## Tools

The project uses:

* Python
* NumPy
* pandas
* Matplotlib
* scikit-learn
* XGBoost

## Future Work

The current project intentionally uses only the six numerical descriptors supplied with the dataset.

A natural extension is to extract richer molecular representations directly from the SMILES strings.

Future work may include:

* Additional physicochemical descriptors generated using **RDKit**
* **Morgan/ECFP fingerprints** representing local molecular substructures
* Comparison of descriptor-based and fingerprint-based models
* Permutation-based feature importance
* Chemical scaffold-based train/test splitting
* Graph-based molecular representations and graph neural networks

These extensions would test whether richer structural information can improve prediction beyond the original six molecular descriptors.

