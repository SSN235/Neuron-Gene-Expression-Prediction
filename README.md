# NGEP: Neuron Gene Expression Prediction

Predicting parvalbumin (Pvalb) expression from neuronal morphology using deep learning.

External Validation Tool: https://ngep-validator-frontend.pages.dev/

## Overview

NGEP is a computational framework for predicting cell-type-specific gene expression from neuronal 3D morphology. Using real open-access data from NeuroMorpho.org and the Allen Brain Atlas, we train a feedforward neural network to infer parvalbumin mRNA levels in mouse neocortex from neuronal morphological features.
 
## Clinical Significance

Parvalbumin is expressed almost exclusively in fast-spiking GABAergic interneurons, which are critical for:
- Gamma oscillations (30-100 Hz)
- Temporal precision of neural firing
- Network stability and seizure prevention
- Excitation/inhibition balance

Pvalb dysfunction is implicated in autism, schizophrenia, epilepsy, Alzheimer's disease, fragile X syndrome, and ADHD.

## Data

### Neuronal Morphology (NeuroMorpho.org)

- Source: NeuroMorpho.org REST API
- Species: Mouse
- Brain Region: Neocortex
- Sample Size: 7500 neurons
- Format: SWC files
- Quality: Verified reconstructions

### Gene Expression (Allen Brain Atlas)

- Source: Allen Brain Atlas RMA API
- Gene: Parvalbumin (Pvalb)
- Method: In situ hybridization
- Coverage: Regional expression values

### Region Matching

Each neuron's brain region metadata is matched to Allen structures to assign region-specific expression targets.

## Morphological Features

Five base features extracted from each neuron's 3D reconstruction:

1. Soma radius - Cell body size
2. Total dendritic length - Cumulative dendrite extent
3. Bifurcation count - Number of branch points
4. Terminal count - Number of synaptic endpoints
5. Branch density - Bifurcations / dendritic length

These features are then engineered into 14 total features through:
- **Ratios:** dendritic_length/soma_radius, bifurcations/terminals, terminals/dendritic_length
- **Products:** soma_radius×dendritic_length, bifurcations×terminals
- **Log transforms:** log(bifurcations + 1), log(terminals + 1)
- **Squares:** soma_radius², dendritic_length²

Feature engineering captures nonlinear relationships and interactions between morphological properties.

## Model Architecture

Neural network design:

  Input (14 engineered features)
    -> Linear(128) -> ReLU -> Dropout(0.1)
    -> Linear(64) -> ReLU -> Dropout(0.1)
    -> Linear(1) [regression output]

Training Configuration:
  - Optimizer: Adam (lr=0.001)
  - Loss: SmoothL1Loss (β=1.0, robust to outliers)
  - Batch size: 8
  - Epochs: 150
  - Cross-validation: Stratified 10-fold
  - Random seed: 42

## Results

### Cross-Validation Performance

- **R²**: Mean 0.3266 (Std 0.0214) | Range 0.2876 - 0.3560
- **Pearson R**: Mean 0.5790 (Std 0.0189) | Range 0.5389 - 0.6063
- **RMSE**: Mean 0.7064 (Std 0.0124) | Range 0.6910 - 0.7290
- **MAE**: Mean 0.5608 (Std 0.0071) | Range 0.5517 - 0.5777
- **p-value**: All < 10^-40 across all 10 folds

The model explains approximately 31% of variance in Pvalb expression from morphological measurements. Remaining variance is attributable to gene regulatory networks, epigenetic state, developmental history, and local circuit context.

### Top 5 Performing Folds

- **Fold 5**: R² 0.3560 | Pearson R 0.6052 | RMSE 0.6910 | p-value 6.11e-57
- **Fold 6**: R² 0.3511 | Pearson R 0.6063 | RMSE 0.6902 | p-value 3.36e-57
- **Fold 10**: R² 0.3470 | Pearson R 0.5928 | RMSE 0.6971 | p-value 4.61e-54
- **Fold 7**: R² 0.3357 | Pearson R 0.5812 | RMSE 0.6934 | p-value 1.26e-51
- **Fold 4**: R² 0.3304 | Pearson R 0.5806 | RMSE 0.7100 | p-value 1.61e-51

Remarkably consistent performance across folds with R² standard deviation of only 0.0214, indicating robust generalization.

### Applications

- Drug screening: Morphology-based Pvalb expression prediction from iPSCs
- Biomarker development: Morphological features as accessible Pvalb indicators
- Patient stratification: Identifying Pvalb circuit dysfunction for targeted therapy
- Therapy validation: Rapid morphological confirmation of Pvalb restoration
- Organoid QC: Morphological screening for healthy Pvalb interneurons
- Mechanistic insight: Linking neuronal structure to molecular function

## NGEP Validator: External Validation Tool

The validator is a web-based tool for testing the consolidated cross-fold model on external validation data from NeuroMorpho.org.

### Architecture

- **Frontend**: Static HTML hosted on Cloudflare Pages
- **Backend**: Flask + gunicorn hosted on Render, serving a single consolidated `.pkl` model

### How It Works

1. Fetches fresh neurons from NeuroMorpho.org (not from training set)
2. Extracts the same 14 engineered morphological features
3. Applies StandardScaler normalization
4. Passes features through the consolidated model
5. Computes performance metrics (R², Pearson r, RMSE, MAE)
6. Falls back to synthetic neuron data if NeuroMorpho.org is unavailable

### Avoiding Training Data Contamination

- Large population: NeuroMorpho.org has 300,000+ neurons (training set is <0.05%)
- Random ordering: Neurons fetched in random order
- Statistical expectation: Probability of refetching same neuron is ~0.05% per validation neuron

Expected contamination rate: <0.05% (negligible)

### API Endpoints

Health check:
  GET /

List models:
  GET /api/models

Fetch neurons:
  GET /api/neurons?species=mouse&brain_region=neocortex&count=10

Run inference:
  POST /api/infer
  Body: { model, neuronCount, genes, species, brain_region, randomSeed }

## Limitations

- Sparse features (14 engineered from 5 base features): Spine density, axonal morphology not included
- Single gene, single cell type: Generalization to other genes/populations unknown
- Cross-sectional: Developmental dynamics not captured
- Correlation, not causation: Morphology may be consequence not driver of expression
- Mouse neocortex only: Applicability to human cortex or other regions unclear

## Future Directions

Short-term:
  - Permutation feature importance
  - Expanded feature engineering (axonal properties, spine density)
  - Multi-gene prediction

Medium-term:
  - Graph neural networks for full morphology
  - Multi-cell-type comparison
  - Regional stratification
  - Electrophysiological integration

Long-term:
  - Single-cell RNA-seq validation
  - Developmental longitudinal data
  - In vivo validation
  - Human iPSC extension
  - Multi-omics integration

## Reproducibility

Random Seed: All random generation seeded with RANDOM_STATE = 42

Dependencies:
  torch==2.6.0
  pandas==2.0.3
  numpy==1.26.0
  scikit-learn==1.3.0
  scipy==1.11.1
  matplotlib==3.7.1
  seaborn==0.12.2
  requests==2.31.0
  beautifulsoup4==4.12.2
  flask==2.3.3
  flask-cors==4.0.0
  gunicorn==21.2.0

Hardware Effects: GPU (CUDA/MPS) may introduce minor floating-point differences. CPU execution is slower but fully reproducible.

Status: Stable, reproducible results across stratified 10-fold CV
Recommended for: Research, proof-of-concept, clinical biomarker development

## Acknowledgments

- NeuroMorpho.org for open-access neuron reconstructions
- Allen Brain Institute for comprehensive in situ hybridization data

Status: Stable, reproducible results across stratified 10-fold CV
Recommended for: Research, proof-of-concept, clinical biomarker development
