# NGEP: Neuron Gene Expression Prediction

**Predicting parvalbumin (Pvalb) expression from neuronal morphology using deep learning.**

Mouse neocortex · NeuroMorpho.org · Allen Brain Atlas · PyTorch

---

## What This Project Does

NGEP asks whether the 3D shape of a neuron — its soma size, dendritic length, and branching pattern — contains enough information to predict how much a cell-type-specific gene is expressed. Using real open-access data and a feedforward neural network, the answer is: **yes, meaningfully, with R² ≈ 0.31 and Pearson R ≈ 0.57** across stratified 10-fold cross-validation.

---

## Results

| Metric | Mean ± Std | All Folds |
|--------|-----------|-----------|
| R² | 0.312 ± 0.021 | 0.257 – 0.340 |
| Pearson R | 0.570 ± 0.015 | 0.534 – 0.589 |
| RMSE | 0.712 ± 0.014 expression units | — |
| MAE | 0.563 ± 0.010 expression units | — |
| p-value | all < 10⁻⁴⁰ | 10/10 significant |

Results were strikingly consistent across all 10 folds — R² std of only 0.021 — indicating robust generalization rather than a lucky split.

---

## Pipeline

```
NGEP_neuron_data_extraction.py   →  150 SWC files + neuron_metadata.csv
NGEP_gene_data_extraction.py     →  expression_by_structure.csv  (Allen Pvalb ISH)
NGEP_feature_extraction.py       →  features.csv  (5 morphological features)
NGEP_data_prep.py                →  features_with_expression.csv  (region-matched targets)
NGEP_model.py                    →  results/  (metrics, plots, model checkpoints)
```

Run scripts in order from the project root. Each script is self-contained and changes to its own directory on startup.

---

## Data Sources

- **NeuroMorpho.org** — SWC morphology files for 150 adult mouse neocortical neurons, fetched via paginated REST API (`species:mouse`, `brain_region` filtered in Python)
- **Allen Brain Atlas** — Pvalb in situ hybridization expression energy per brain structure, fetched via RMA query API (product ID 1 = adult mouse)

---

## Features

Five morphological features are extracted from each SWC file:

| Feature | Definition |
|---------|-----------|
| Soma radius | Sum of radii of all soma-type (type=1) nodes |
| Total dendritic length | Cumulative Euclidean distance between consecutive dendrite nodes |
| Bifurcation count | Number of nodes with ≥ 2 children |
| Terminal count | Number of leaf nodes (no children) |
| Branch density | Bifurcations / (dendritic length + 1e-6) |

---

## Model

A two-layer feedforward network in PyTorch:

```
Input (5) → Linear(128) → BatchNorm → ReLU → Dropout(0.1)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.1)
          → Linear(1)   [regression output]
```

- **Optimizer:** Adam, lr=0.001
- **Loss:** MSE
- **Epochs:** 150, batch size 8
- **CV:** Stratified 10-fold (stratified by brain region)
- **Scaler:** StandardScaler fit on training fold only (no data leakage)
- **Device:** MPS (Apple Silicon) / CUDA / CPU auto-detected

---

## Key Design Decisions

**Region-specific expression targets:** Averaging Pvalb expression across all brain structures assigns every neuron the same target value, eliminating learning signal. Instead, each neuron's `brain_region` string is parsed into anatomical keywords that are matched to Allen structure name patterns. Mean expression_energy across matched structures is the neuron's target.

**Python-side brain region filtering:** The NeuroMorpho API stores `brain_region` as an array; server-side filtering is unreliable. Neurons are filtered in Python with `if "neocortex" in brain_regions`.

**SWC URL resolution via HTML scraping:** SWC download URLs are not directly available from the API; they are parsed from each neuron's HTML info page using BeautifulSoup.

---

## Outputs

```
data/
  neuromorpho/          ← 150 .swc files + neuron_metadata.csv
  allen/                ← expression_by_structure.csv
  features.csv          ← 150 neurons × 8 columns
  features_with_expression.csv  ← final training-ready dataset

models/
  fold_0_best.pt … fold_9_best.pt   ← PyTorch state dicts

results/
  cross_validation_metrics.csv
  cv_r2_scores.png
  predicted_vs_actual.png
  residuals.png
  learning_curves/
    all_folds_learning_curves.png
    fold_01_learning_curve.png … fold_10_learning_curve.png
```

---

## Requirements

```
torch, numpy, pandas, scipy, sklearn, matplotlib, requests, beautifulsoup4, networkx
```

---

## Scope & Limitations

This project is intentionally scoped: **adult mouse neocortex only, Pvalb only**, with 150 neurons and 5 features. This scope was chosen to produce a complete, validated pipeline within a fixed project timeline.

Limitations include the small dataset, sparse feature set (no spine density, axonal properties, or electrophysiology), and the correlation-not-causation nature of the morphology-expression link. See the write-up PDF for full discussion and future directions.

---

## Citation

If referencing this project, please also cite the underlying data sources:

- Ascoli et al. (2007). NeuroMorpho.Org. *Journal of Neuroscience*, 27(35).
- Lein et al. (2007). Allen Brain Atlas. *Nature*, 445(7124).
