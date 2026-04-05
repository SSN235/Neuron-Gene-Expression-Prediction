# NGEP: Neuron Gene Expression Prediction

**Predicting parvalbumin expression from neuron shape using deep learning.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Here's the basic idea: neurons have wildly different shapes. Some are compact and bushy, others are spread out. Some branch like trees, others are simple. We wondered—does that shape tell you something about what genes the neuron expresses?

This project uses real 3D neuron reconstructions from NeuroMorpho.org and gene expression data from the Allen Brain Atlas to train a neural network on a simple question: can you predict how much parvalbumin a neuron makes just by measuring its morphology?

The answer is yes—and it's more predictive than I expected.

**Current results (10-fold CV):**
- R² = 0.3956 ± 0.027 (explains ~40% of the variance)
- Pearson r = 0.6322 ± 0.020 (solid correlation)
- MAE = 0.5073 ± 0.013
- RMSE = 0.6691 ± 0.016
- Every single fold came back significant (p < 10⁻⁵¹)

Not perfect, but definitely real. The consistent performance across all 10 folds means we're not overfitting—the model actually learned something about the morphology-expression relationship.

---

## Quick Start

### Try the Interactive Validator

Before running the full pipeline, test the model on new validation data using the **NGEP Validator Tool**:

🌐 **[NGEP Validator](https://ngep-validator-frontend.pages.dev/)**

The validator lets you:
- Pick how many neurons to test (1–1000+)
- Automatically grab fresh data from neuromorpho.org (randomized, not from training)
- Get real-time predictions with performance metrics
- See predicted vs actual expression in a scatter plot
- Check if the model generalizes to new data

**Backend:** Render | **Frontend:** Cloudflare Pages

---

### Installation & Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/NGEP.git
cd NGEP

# Virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn scipy matplotlib seaborn beautifulsoup4 requests
```

### Run the Pipeline

```bash
# Step 1: Download neuron morphologies from NeuroMorpho.org
python NGEP_neuron_data_extraction.py

# Step 2: Download gene expression from Allen Brain Atlas
python NGEP_gene_data_extraction.py

# Step 3: Extract morphological features from SWC files
python NGEP_feature_extraction.py

# Step 4: Match brain regions to expression values
python NGEP_data_prep.py

# Step 5: Train with stratified 10-fold cross-validation
python NGEP_model.py
```

Results go to `results/` and trained models to `models/`.

---

## Results

### Cross-Validation Performance

Here's what we got across all 10 folds:

| Metric | Mean ± Std | Range | Notes |
|--------|-----------|-------|-------|
| **R²** | **0.3956 ± 0.0270** | 0.3400 – 0.4447 | Pretty consistent |
| **Pearson R** | **0.6322 ± 0.0200** | 0.5899 – 0.6674 | Strong correlation |
| **RMSE** | **0.6691 ± 0.0155** | 0.6424 – 0.7012 | Stable predictions |
| **MAE** | **0.5073 ± 0.0128** | 0.4817 – 0.5251 | Off by ~0.5 units on average |
| **p-value** | **all < 10⁻⁵¹** | 10/10 folds | Definitely not random |

**What this means:** The model explains about 40% of the variance in parvalbumin expression using only morphology. That's not bad considering there are probably tons of other things that matter (epigenetics, development, activity, neuromodulators, etc.). But for just looking at shape? Pretty solid.

### Top Performing Folds

| Fold | R² | Pearson R | RMSE | p-value |
|------|-----|-----------|------|---------|
| **9** | **0.4447** | **0.6674** | **0.6424** | **5.98e-73** |
| **10** | **0.4252** | **0.6528** | **0.6540** | **7.90e-69** |
| **4** | **0.4048** | **0.6375** | **0.6694** | **7.49e-65** |
| **7** | **0.4066** | **0.6393** | **0.6554** | **2.44e-65** |
| **6** | **0.3922** | **0.6275** | **0.6680** | **2.60e-62** |

Fold 9 crushed it. Pretty interesting that it's so consistent though—the worst fold (Fold 2) still got R² = 0.34, so we didn't get weird unlucky splits.

---

## NGEP Validator: External Validation Tool

### What It Does

The validator is a web tool that tests the trained model on completely new neurons from NeuroMorpho.org. This lets you independently verify that the model actually generalizes instead of just memorizing the training data.

🌐 **Live Tool:** [https://ngep-validator-frontend.pages.dev/](https://ngep-validator-frontend.pages.dev/)

### How It Works

The validator:
1. **Fetches fresh neurons** from NeuroMorpho.org using randomized API pages (so it avoids the training set)
2. **Filters for neocortex only** (same as training population)
3. **Extracts the same 14 engineered features** we used in training
4. **Runs the ensemble of 10 trained models** and averages predictions
5. **Compares to ground truth** using a static lookup table of brain regions → expression values
6. **Computes metrics** (R², Pearson r, RMSE, MAE) and shows you everything

### Key Implementation Details

**Why randomized page fetching?** NeuroMorpho.org has >300,000 neurons total. We used ~7,500 (2.5%) for training. By randomizing which pages we fetch from, we minimize overlap with the training set.

**Why a static expression map?** To avoid systematic drift from re-querying a truncated dataset at validation time. We pre-computed 129 brain_region → expression_energy mappings during training using the full Allen Brain Atlas ISH data, and the validator uses those same values. That way, the only thing being tested is whether the model can handle *unseen neuron morphologies*.

**Feature consistency:** The validator uses the exact same StandardScaler, feature engineering, and SWC parsing logic as the training pipeline. Everything is identical except the neuron reconstructions themselves.

### How to Use It

#### Quick test (10 neurons, ~30 seconds):
1. Go to [NGEP Validator](https://ngep-validator-frontend.pages.dev/)
2. Set slider to 10
3. Hit "Fetch Data & Validate"
4. Check the results—should see R² ≈ 0.30-0.40

#### Serious validation (100 neurons, 2-3 minutes):
1. Set slider to 100
2. Click "Fetch Data & Validate"
3. Download results as CSV
4. Import into R/Python for secondary analysis
5. Compare to your training results

#### Monthly monitoring:
1. Run with 50 neurons every month
2. Track whether performance stays stable over time
3. If it drops, retrain on expanded dataset
4. Good way to catch model drift

### API Endpoints (For Developers)

```
POST /api/fetch-neurons
{
  "count": 50,
  "species": "mouse",
  "brain_region": "neocortex",
  "randomize": true
}

Response:
{
  "neurons": [
    {"name": "neuron_123", "features": [2.5, 1000, 50, 100, 0.05]},
    ...
  ],
  "count": 50
}
```

```
POST /api/predict
{
  "features": [[2.5, 1000, 50, 100, 0.05], ...],
  "model_version": "1.0"
}

Response:
{
  "predictions": [6.2, 6.8, 5.9, ...],
  "metadata": {
    "model": "NGEP v1.0 (10-fold ensemble)",
    "n_predictions": 50
  }
}
```

```
POST /api/evaluate
{
  "actual": [6.5, 7.0, 5.8, ...],
  "predicted": [6.2, 6.8, 5.9, ...]
}

Response:
{
  "metrics": {
    "r2": 0.31,
    "pearson_r": 0.56,
    "rmse": 0.71,
    "mae": 0.56
  }
}
```

---

## Dataset

### Neuronal Morphology (NeuroMorpho.org)

- **Source:** NeuroMorpho.org REST API
- **Species:** Mouse
- **Brain Region:** Neocortex
- **Total Neurons:** 7,499
- **File Format:** SWC (standard morphology format)
- **Quality:** Verified reconstructions from multiple labs

### Gene Expression (Allen Brain Atlas)

- **Source:** Allen Brain Atlas RMA API
- **Gene:** Parvalbumin (Pvalb)
- **Method:** In situ hybridization (ISH)
- **Species:** Mouse, adult
- **Metric:** Expression energy (normalized intensity)
- **Coverage:** ~70 neocortical substructures

### Region Matching

Each neuron in the dataset has brain region metadata like "neocortex, occipital, layer 5". We match this to Allen structures using keyword-based lookups:
- Cortical areas: occipital, somatosensory, motor, temporal, parietal, prefrontal, etc.
- Layers: 1–6 (with special handling for layer 2/3 and 6b)
- Substructures: primary/secondary, orbital, insula, cingulate, etc.

This gives us enough variance in the target (~0.49) to actually train on.

---

## Features

### The Five Morphological Measurements

These are the basic features we extract from each neuron's SWC file:

| Feature | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Soma radius** | Total radius of soma nodes | Cell body size; relates to metabolism and electrical properties |
| **Total dendritic length** | Sum of all distances between dendrite nodes | How much surface area for receiving inputs |
| **Bifurcation count** | How many times dendrite branches | Structural complexity; elaboration of the tree |
| **Terminal count** | Number of leaf nodes (endpoints) | Potential number of output synapses |
| **Branch density** | Bifurcations / dendritic length | How compact the branching is; local vs. distributed structure |

**Why these five?** Parvalbumin+ interneurons are known for compact, heavily branched dendrites. These measurements capture both overall size and the branching pattern—basically the morphological signature of fast-spiking interneurons.

---

## Model Architecture

### Feature Engineering (5 → 14 Features)

We took the 5 base features and derived 9 more to capture nonlinear relationships:

**Ratios:**
- dendritic_length / soma_radius
- bifurcations / terminals
- terminals / dendritic_length

**Products:**
- soma_radius × dendritic_length
- bifurcations × terminals

**Log Transforms:**
- log(bifurcations + 1)
- log(terminals + 1)

**Squares:**
- soma_radius²
- dendritic_length²

These transformations help the network learn interactions and nonlinear effects.

### Neural Network

```
Input (14 features)
    ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(1) → Output (expression energy)
```

**Why this design:**
- Two hidden layers is enough for the problem size
- Batch normalization stabilizes training and helps generalization
- Light dropout (0.1) prevents overfitting without hurting learning
- Linear output for unbounded regression

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | SmoothL1Loss (robust to outliers) |
| Batch size | 8 |
| Epochs | 150 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=10) |

### Cross-Validation

- **Method:** Stratified 10-fold (stratified by brain region)
- **Scaler:** StandardScaler fitted on each training fold only (prevents data leakage)
- **Random seed:** 42
- **Fold size:** ~6,750 training, ~750 test per fold

---

## Project Structure

```
NGEP/
├── README.md
├── NGEP_writeup.pdf
│
├── Data Extraction
├── NGEP_neuron_data_extraction.py
├── NGEP_gene_data_extraction.py
│
├── Feature Engineering
├── NGEP_feature_extraction.py
├── NGEP_data_prep.py
│
├── Model Training
├── NGEP_model.py
│
├── data/
│   ├── neuromorpho/
│   │   ├── neuron_metadata.csv
│   │   └── *.swc
│   ├── allen/
│   │   └── expression_by_structure.csv
│   ├── features.csv
│   └── features_with_expression.csv
│
├── results/
│   ├── cross_validation_metrics.csv
│   ├── cv_r2_scores.png
│   ├── predicted_vs_actual.png
│   ├── residuals.png
│   └── learning_curves/
│       └── [fold_01-10_learning_curve.png]
│
└── models/
    ├── fold_0_best.pt
    ├── fold_1_best.pt
    └── ... [fold 2-9]
```

---

## Why Parvalbumin Matters Clinically

### The Biology

Parvalbumin is expressed almost exclusively in fast-spiking GABAergic interneurons. These are the brain's "rhythm keepers"—they generate synchronized oscillations and maintain inhibitory control. They're critical for:

- **Gamma oscillations** (30–100 Hz): linked to attention, memory, sensory processing
- **Temporal precision:** coordinating exact spike timing across populations
- **E/I balance:** preventing seizures and hyperexcitation
- **Network stability:** controlling the flow of information

### Why It's Broken in Disease

**Autism & ASD:** Reduced Pvalb+ function correlates with social deficits. Stimulating these neurons rescues social behavior in mouse models.

**Schizophrenia:** Selective loss of Pvalb+ cells in prefrontal cortex. Disrupted gamma oscillations predict cognitive symptoms.

**Epilepsy:** Less Pvalb expression → less inhibition → seizures. Genetic variants affecting Pvalb are risk factors.

**Alzheimer's:** Pvalb+ interneurons degenerate early. Their loss correlates with memory deficits.

**Fragile X:** Excessive excitation + impaired Pvalb maturation. Restoring these cells rescues behavioral symptoms.

**Parkinson's:** Altered Pvalb circuits in basal ganglia contribute to motor dysfunction.

**ADHD:** Pvalb interneurons in prefrontal cortex support sustained attention. Dysfunction → attention deficits.

### Clinical Applications

1. **Drug screening:** Pharmaceutical companies developing therapies to enhance Pvalb expression or function could use this model to rapidly screen neurons from patient-derived cells

2. **Biomarkers:** Predict Pvalb levels from microscopy without expensive genomic assays

3. **Patient stratification:** Use morphological imaging to identify patients with Pvalb circuit dysfunction

4. **Treatment validation:** Quickly verify whether new ASD, schizophrenia, or epilepsy drugs restore Pvalb interneuron function

5. **Quality control:** Screen organoids and iPSC-derived neurons for healthy Pvalb interneurons before clinical use

6. **Understanding mechanisms:** Figure out which morphological features matter most for proper Pvalb expression

---

## What We Actually Learned

The model explains ~40% of variance. That leaves 60% unexplained. Why? Because morphology is only part of the story:

- **Epigenetics:** DNA methylation, chromatin state
- **Transcription factors:** Which regulatory proteins are active
- **Development:** Whether the neuron is mature, immature, or still developing
- **Circuit context:** What other neurons it's connected to
- **Neuromodulation:** Dopamine, serotonin, acetylcholine, etc.
- **Electrophysiology:** Firing patterns, input resistance, membrane properties
- **Metabolic state:** Energy availability, growth state

The fact that morphology alone gets us to 40% is actually pretty interesting. It suggests that shape and gene expression are genuinely linked, not just random correlation.

---

## Limitations

**What we did well:**
- Used real data throughout (no synthetic data)
- Region-specific targets (each neuron matched to its brain region's expression)
- Rigorous cross-validation with proper scaling (no data leakage)
- Highly consistent results across folds (std = 0.027)
- All results are statistically significant (p < 10⁻⁵¹)
- Full reproducibility (code + data sources provided)

**What we didn't do:**
- Dataset is moderate size (7,499 neurons; bigger would help)
- Only measured 5 base features (spine density, axonal properties, electrophysiology not included)
- Only one gene, one cell type (generalization unknown)
- Cross-sectional snapshot (no developmental dynamics)
- Correlation, not causation (morphology might be a consequence, not cause, of expression)
- Mouse neocortex only (applicability to human brain or other regions unclear)

---

## Future Work

### Short-term (doable soon)
- Permutation feature importance (which of the 5 features matters most?)
- More engineered features (soma-to-dendrite ratios, spine density, path asymmetry)
- Larger dataset (500–1,000 neurons; NeuroMorpho.org has 300k+ available)
- Multi-gene prediction (GAD1, GFAP, other interneuron markers)

### Medium-term (6–12 months)
- Graph neural networks (learn from full morphology structure instead of hand-crafted features)
- Compare interneurons vs. pyramidal neurons (is the morphology-expression link universal?)
- Per-cortical-area models (V1 vs. M1 vs. PFC might have different relationships)
- Electrophysiology integration (add measured firing rates, input resistance, etc.)

### Long-term (1–2 years)
- Cross-validate with single-cell RNA-seq data (MERFISH, 10x Genomics)
- Developmental time series (track morphology-expression changes from birth to adulthood)
- In vivo validation (measure expression + morphology in the same neurons using 2-photon microscopy)
- Human translation (test on iPSC-derived neurons and patient tissue)
- Multi-omics integration (proteomics, phosphoproteomics, ATAC-seq for mechanisms)

---

## Configuration

All hyperparameters at the top of `NGEP_model.py`:

```python
DATA_FILE = 'data/features_with_expression.csv'
USE_GPU = True
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT = 0.1
NUM_FOLDS = 10
RANDOM_STATE = 42
```

**For larger datasets (500+ neurons), consider:**
```python
HIDDEN_1 = 256
HIDDEN_2 = 128
HIDDEN_3 = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0005
```

---

## Reproducibility

Everything is seeded with `RANDOM_STATE = 42`. Results should be perfectly reproducible on CPU. GPU can introduce minor floating-point differences.

To get CPU-only runs, set `USE_GPU = False` in `NGEP_model.py`.

Exact versions:
```
torch==2.0.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.1
seaborn==0.12.2
```
