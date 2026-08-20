# ESOL Aqueous Solubility Prediction with Machine Learning

An independent supervised learning study of aqueous molecular solubility using the **ESOL (Delaney) dataset**.

The project progresses from exploratory data analysis and simple regression baselines to tree ensembles, chemistry-informed feature engineering with RDKit, SHAP interpretability, chemistry-aware scaffold splitting, and a baseline graph neural network.

The main goal is not only to obtain a low test error, but to understand how **molecular representation and evaluation strategy affect apparent model performance**.

Note: Solubility is a real HARD problem

---

## TL;DR

| Setting                                           | Test RMSE |
| ------------------------------------------------- | --------: |
| Mean predictor                                    |     2.174 |
| Provided ESOL prediction                          |     0.956 |
| Linear Regression, 6 descriptors                  |     1.199 |
| XGBoost, 6 descriptors                            |     0.874 |
| **XGBoost, 6 descriptors + logP**                 | **0.741** |
| **XGBoost, 6 descriptors + logP, scaffold split** | **0.940** |
| Baseline 2-layer GCN, scaffold split              |     1.220 |

### Main results

* Adding a single chemically motivated feature, **logP**, reduced random-split XGBoost RMSE from **0.874 to 0.741**.
* Moving from a random split to a **Bemis–Murcko scaffold split** increased RMSE from **0.741 to 0.940**, demonstrating that prediction on structurally unfamiliar molecules is substantially harder.
* Under the same scaffold split, the descriptor-based XGBoost model outperformed a baseline 2-layer GCN: **0.940 vs 1.220 RMSE**.
* SHAP analysis identified **logP as the dominant predictor**, followed by Polar Surface Area, Molecular Weight, and Number of Rings.

---

# 1. Problem

For each molecule, the objective is to predict experimentally measured aqueous solubility:

$$
y = \log_{10}(S),
$$

where (S) is the solubility in mol/L.

This is a **supervised regression problem**.

The dataset contains **1,128 molecules** represented by molecular descriptors and SMILES strings.

The target column is:

```text
measured log solubility in mols per litre
```

---

# 2. Dataset

The project uses the MoleculeNet version of the Delaney ESOL dataset.

### Dataset statistics

* Number of molecules: **1,128**
* Missing values: **0**
* Duplicate SMILES: **5**
* Target range: approximately **−11.60 to +1.58 logS**
* Mean target: approximately **−3.05**
* Target standard deviation: approximately **2.10**

### Initial features

The first set of models used six supplied numerical descriptors:

1. Molecular Weight
2. Polar Surface Area
3. Number of Rings
4. Number of Rotatable Bonds
5. Number of H-Bond Donors
6. Minimum Degree

The dataset also contains:

* `Compound ID` — molecule identifier
* `smiles` — molecular structure representation
* `ESOL predicted log solubility in mols per litre` — supplied ESOL prediction used as a reference

---

# 3. Exploratory Data Analysis

Before modeling, the dataset was inspected for:

* Data types and feature ranges
* Missing values
* Duplicate observations
* Target distribution
* Potential outliers
* Relationships between molecular descriptors and solubility

## Low-solubility outliers

Using the (1.5\times IQR) criterion, **17 molecules** were identified as target outliers.

All were on the extremely low-solubility side.

| Property                  |   Mean |
| ------------------------- | -----: |
| Molecular Weight          | 339.81 |
| Number of Rings           |   3.06 |
| Number of Rotatable Bonds |   3.00 |
| Number of H-Bond Donors   |   0.06 |
| Polar Surface Area        |   6.31 |
| Measured logS             |  −8.91 |

These molecules tend to combine relatively large molecular size with very low polarity, consistent with poor aqueous solubility.

The observations were **retained** because statistical outliers are not necessarily erroneous measurements and may represent chemically meaningful examples.

---

# 4. Initial Modeling: Six Descriptors

The data were first divided using an **80/20 random train/test split** with:

```python
random_state = 42
```

Hyperparameters were selected using **5-fold cross-validation on the training set**.

The held-out test set was used for final evaluation.

Models included:

* Mean baseline
* Linear Regression
* Ridge Regression
* Polynomial Ridge Regression
* Random Forest
* Gradient Boosting
* XGBoost

## Results

| Model                    | Features      | Test RMSE | Test MAE | Test R² |
| ------------------------ | ------------- | --------: | -------: | ------: |
| Mean baseline            | —             |     2.174 |    1.732 |   0.000 |
| Provided ESOL prediction | —             |     0.956 |    0.708 |   0.807 |
| Linear Regression        | 6 descriptors |     1.199 |    0.892 |   0.696 |
| Ridge                    | 6 descriptors |     1.199 |    0.892 |   0.696 |
| Polynomial Ridge         | 6 descriptors |     1.057 |        — |   0.764 |
| Random Forest            | 6 descriptors |     0.875 |        — |   0.838 |
| Gradient Boosting        | 6 descriptors |     0.876 |        — |   0.838 |
| XGBoost                  | 6 descriptors |     0.874 |    0.597 |   0.838 |

The three tree ensembles produced nearly identical held-out performance:

$$
\text{Random Forest}
\approx
\text{Gradient Boosting}
\approx
\text{XGBoost}.
$$

All three substantially outperformed the linear and polynomial models.

This suggests that nonlinear relationships and feature interactions are important when predicting solubility from these descriptors.

---

# 5. Chemistry-Informed Feature Engineering

Rather than continuing to tune models on the same six descriptors, the next experiment changed the **molecular representation**.

A single new feature was derived directly from SMILES using RDKit:

### Crippen logP

logP measures hydrophobicity through the octanol/water partition coefficient:

$$
\log P =
\log_{10}
\left(
\frac{[\text{solute}]*{\text{octanol}}}*
*{[\text{solute}]*{\text{water}}}
\right).
$$

Higher logP generally corresponds to greater hydrophobicity and therefore often lower aqueous solubility.

The feature matrix changed from:

$$
X_6 =
[\mathrm{MW, PSA, Rings, RotBonds, HBD, MinDegree}]
$$

to:

$$
X_7 =
[X_6,\log P].
$$

Everything else—the random split, model family, and evaluation procedure—was kept fixed so that the effect of adding logP could be isolated.

## Effect of adding logP

| Model             | Features                 | Test RMSE |  Test MAE |   Test R² |
| ----------------- | ------------------------ | --------: | --------: | --------: |
| Linear Regression | 6 descriptors            |     1.199 |     0.892 |     0.696 |
| Linear Regression | 6 descriptors + logP     |     1.064 |     0.811 |     0.761 |
| XGBoost           | 6 descriptors            |     0.874 |     0.597 |     0.838 |
| **XGBoost**       | **6 descriptors + logP** | **0.741** | **0.519** | **0.884** |

For XGBoost:

$$
0.874 \rightarrow 0.741
$$

corresponding to approximately a **15% reduction in test RMSE** from adding a single chemically motivated feature.

This was a larger improvement than further tuning among the original six descriptors.

---

# 6. Model Interpretability

## XGBoost gain importance

For the 7-feature XGBoost model:

| Feature                   | XGBoost Gain |
| ------------------------- | -----------: |
| **logP**                  |    **0.585** |
| Molecular Weight          |        0.100 |
| Polar Surface Area        |        0.087 |
| Number of Rings           |        0.086 |
| Number of Rotatable Bonds |        0.048 |
| Number of H-Bond Donors   |        0.048 |
| Minimum Degree            |        0.048 |

Adding logP substantially changed the feature-importance picture.

Among the original six descriptors, Molecular Weight had been the strongest feature. Once a direct hydrophobicity descriptor was introduced, logP became dominant.

---

# 7. SHAP Analysis

SHAP was used to analyze how individual features influence individual predictions.

For tree models:

$$
\hat y_i =
E[f(X)]
+
\sum_j \phi_{ij},
$$

where (\phi_{ij}) is the SHAP contribution of feature (j) to prediction (i).

The numerical additivity check reproduced model predictions to approximately machine precision.

## Global SHAP importance

| Feature | Mean (|\text{SHAP}|) |
|---|---:|
| **logP** | **1.335** |
| Polar Surface Area | 0.321 |
| Molecular Weight | 0.236 |
| Number of Rings | 0.215 |
| Number of Rotatable Bonds | 0.095 |
| Number of H-Bond Donors | 0.049 |
| Minimum Degree | 0.022 |

The ranking is:

$$
\boxed{
\log P

>

PSA

>

MW

>

Rings

>

Rotatable

>

HBD

>

MinimumDegree
}
$$

logP shifts predictions by approximately **1.34 logS units on average in magnitude**, making hydrophobicity the dominant predictive signal in this model.

### Gain vs SHAP

XGBoost gain and SHAP agree that logP is overwhelmingly important, but they rank PSA and Molecular Weight differently.

This illustrates that the methods answer different questions:

* **Gain** measures how useful a feature was for reducing training loss through tree splits.
* **SHAP** measures how strongly a feature changes actual model predictions.

---

# 8. Chemistry-Aware Generalization: Scaffold Split

Random train/test splitting can place structurally similar molecules in both sets.

To create a harder generalization test, molecules were grouped using the **Bemis–Murcko scaffolds**.

All molecules belonging to a scaffold were assigned entirely to either training or test data.

This asks a more difficult question:

> Can the model predict solubility for molecules built around structural frameworks that were not represented in the training set?

## Random vs scaffold split

Using the same 7-feature XGBoost representation:

| Metric    | Random Split | Scaffold Split | Change |
| --------- | -----------: | -------------: | -----: |
| Test RMSE |    **0.741** |      **0.940** | +26.8% |
| Test R²   |    **0.884** |      **0.793** | −10.3% |

The performance drop demonstrates that random splitting gives a considerably easier prediction problem.

The scaffold-split result should therefore be viewed as a more demanding estimate of generalization to structurally unfamiliar molecules.

---

# 9. SHAP Under Scaffold Split

SHAP was repeated for the scaffold-split model.

| Feature                   | Random Split | Scaffold Split |
| ------------------------- | -----------: | -------------: |
| **logP**                  |    **1.335** |      **1.611** |
| Polar Surface Area        |        0.321 |          0.363 |
| Molecular Weight          |        0.236 |          0.363 |
| Number of Rings           |        0.215 |          0.228 |
| Number of Rotatable Bonds |        0.095 |          0.091 |
| Number of H-Bond Donors   |        0.049 |          0.091 |
| Minimum Degree            |        0.022 |          0.023 |

logP remained the dominant feature and its average SHAP magnitude increased under the harder scaffold split.

This is consistent with the model relying more heavily on general physicochemical information when predicting molecules with unfamiliar structural frameworks.

---

# 10. Baseline Graph Neural Network

The final experiment replaced hand-crafted molecular descriptors with a representation learned directly from molecular graphs.

A baseline Graph Convolutional Network was implemented using PyTorch Geometric.

### Architecture

```text
GCNConv
→ ReLU
→ GCNConv
→ ReLU
→ Global Mean Pooling
→ MLP
→ Predicted logS
```

The network used:

* 2 graph convolution layers
* Hidden dimension: 64
* Adam optimizer
* Learning rate: (10^{-3})
* Weight decay: (10^{-5})
* Batch size: 32
* 100 training epochs
* Approximately 9,000 trainable parameters

The same scaffold split was used for a direct comparison with XGBoost.

## XGBoost vs GCN

| Model        | Representation                         | Test RMSE |   Test R² |
| ------------ | -------------------------------------- | --------: | --------: |
| **XGBoost**  | 7 physicochemical descriptors          | **0.940** | **0.793** |
| Baseline GCN | Learned molecular graph representation |     1.220 |     0.652 |

The baseline GCN did **not** outperform descriptor-based XGBoost.

This result demonstrates that learned representations do not automatically outperform engineered molecular descriptors, particularly in a small-data setting.

### Possible limitations of the baseline GCN

1. **Limited bond information**

   Vanilla `GCNConv` primarily propagates node representations through graph connectivity and does not directly exploit the full bond-feature vector used to construct the molecular graph.

2. **Simple global mean pooling**

   Mean pooling averages atomic embeddings and does not explicitly preserve molecular size information.

3. **Small training dataset**

   Only about 900 training molecules are available, which is limited for learning a molecular representation from scratch.

More chemistry-aware message-passing architectures such as GINE, D-MPNN, or Chemprop-style models would be natural future extensions.

---

# 11. Key Findings

### 1. Feature representation mattered more than additional model tuning

Random Forest, Gradient Boosting, and XGBoost all reached approximately:

$$
RMSE \approx 0.87
$$

using the original six feature descriptors.

Adding one physically motivated feature—logP—reduced XGBoost RMSE to:

$$
\boxed{0.741}.
$$

---

### 2. Hydrophobicity is the dominant predictive signal

Both XGBoost gain and SHAP identified logP as the most important feature by a large margin.

This agrees with the chemical expectation since the aqueous solubility depends strongly on the molecule's affinity for polar versus nonpolar environments.

---

### 3. Random-split performance can be optimistic

Performance degraded from:

$$
0.741
\rightarrow
0.940
$$

when switching from a random split to a scaffold split.

Models therefore generalized less effectively to molecules containing unfamiliar structural frameworks.

---

### 4. Model interpretation depends on the importance method

Random Forest impurity importance, XGBoost gain, permutation importance, and SHAP do not measure exactly the same concept.

SHAP was particularly useful because it provided:

* Global feature ranking
* Prediction-space units
* Direction of feature effects
* Per-molecule explanations

---

### 5. Better neural-network architecture is not equivalent to more depth

A baseline two-layer GCN underperformed XGBoost.

Simply adding GCN layers is unlikely to address the main representational limitations.

A more meaningful next step would be a bond-aware message-passing model rather than simply making the vanilla GCN deeper.

---

# 12. Figures

All final figures are stored in `figures/`.

### &#x20;

---

# 13. Notebook Structure

Run the notebooks in the following order:

```text
1. Data_processing.ipynb
2. Model_testing.ipynb
3. Feature_engineering.ipynb
4. Scaffold_split.ipynb
5. GNN_pytorch.ipynb
```

### `Data_processing.ipynb`

* Dataset inspection
* Missing values
* Duplicate analysis
* Target distribution
* Outlier analysis
* Basic chemical interpretation

### `Model_testing.ipynb`

* Train/test split
* Mean baseline
* Linear Regression
* Ridge
* Polynomial Ridge
* Random Forest
* Gradient Boosting
* XGBoost
* GridSearchCV
* Model comparison
* Permutation importance

### `Feature_engineering.ipynb`

* Generate Crippen logP from SMILES with RDKit
* Compare 6-feature and 7-feature models
* Retrain XGBoost
* SHAP global analysis
* SHAP local explanations

### `Scaffold_split.ipynb`

* Generate Bemis–Murcko scaffolds
* Construct scaffold-aware train/test split
* Retrain the 7-feature XGBoost model
* Compare random and scaffold splits
* SHAP analysis under scaffold split

### `GNN_pytorch.ipynb`

* Convert SMILES into molecular graphs
* Train a baseline 2-layer GCN
* Evaluate under the same scaffold split
* Compare learned and engineered molecular representations

---

# 14. Project Structure

```text
ESOL/
│
├── README.md
├── delaney-processed.csv
│
├── Data_processing.ipynb
├── Model_testing.ipynb
├── Feature_engineering.ipynb
├── Scaffold_split.ipynb
├── GNN_pytorch.ipynb
│
```

---

# 15. Limitations

### Small dataset

ESOL contains only 1,128 molecules.

This limits the amount of information available for training highly parameterized learned representations such as GNNs.

### Experimental uncertainty

Solubility measurements can depend on experimental protocol, pH, ionization state, temperature, and data source.

This places practical limits on predictive accuracy.

### No explicit solid-state/crystal information

The models operate on 2D molecular representations.

Aqueous solubility also depends on properties such as crystal packing and lattice stability around the solvent molecules, which are not explicitly represented.

### Scaffold splitting is not the only realistic evaluation

Scaffold splitting provides a useful structural generalization test, but other evaluation regimes—such as temporal splits or explicit out-of-distribution datasets—can provide additional information about deployment performance.

### Baseline GCN

The GCN is intentionally simple and should not be interpreted as representing the performance limit of graph neural networks for solubility prediction.

---

# 16. Future Work

Potential extensions include:

* Full RDKit physicochemical descriptor set
* Morgan / ECFP molecular fingerprints
* Comparison of fingerprints vs descriptors
* Bond-aware GNN architectures such as GINE
* Directed message-passing networks such as Chemprop
* Pretrained molecular representations
* Larger solubility datasets such as AqSolDB
* Scaffold-aware cross-validation during hyperparameter tuning
* Prediction uncertainty and calibration
* External validation on non-overlapping molecular datasets

---

# 17. Reproducibility

All train/test splits and stochastic models use:

```python
random_state = 42
```

Core packages include:

```text
Python
NumPy
pandas
scikit-learn
RDKit
XGBoost
SHAP
PyTorch
PyTorch Geometric
```

---

# 18. References

1. Delaney, J. S.
   *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.*
   Journal of Chemical Information and Computer Sciences, **44**, 1000–1005 (2004).
   DOI: 10.1021/ci034243x

2. Wu, Z. et al.
   *MoleculeNet: A Benchmark for Molecular Machine Learning.*
   Chemical Science, **9**, 513–530 (2018).
   DOI: 10.1039/C7SC02664A

3. Lundberg, S. M. & Lee, S.-I.
   *A Unified Approach to Interpreting Model Predictions.*
   NeurIPS (2017).

4. Chen, T. & Guestrin, C.
   *XGBoost: A Scalable Tree Boosting System.*
   KDD (2016).
   DOI: 10.1145/2939672.2939785

5. Kipf, T. N. & Welling, M.
   *Semi-Supervised Classification with Graph Convolutional Networks.*
   ICLR (2017).

6. Fey, M. & Lenssen, J. E.
   *Fast Graph Representation Learning with PyTorch Geometric.*
   ICLR Workshop (2019).

7. Sorkun, M. C., Khetan, A. & Er, S.
   *AqSolDB, a Curated Reference Set of Aqueous Solubility and 2D Descriptors for a Diverse Set of Compounds.*
   Scientific Data, **6**, 143 (2019).
   DOI: 10.1038/s41597-019-0151-1

8. Yang, K. et al.
   *Analyzing Learned Molecular Representations for Property Prediction.*
   Journal of Chemical Information and Modeling, **59**, 3370–3388 (2019).
   DOI: 10.1021/acs.jcim.9b00237

9. Gilmer, J. et al.
   *Neural Message Passing for Quantum Chemistry.*
   ICML (2017).

