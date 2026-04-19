# NGEP: Neuron Gene Expression Prediction

Predicting parvalbumin (Pvalb) expression from neuronal morphology using deep learning.

External Validation Tool: https://ngep-validator-frontend.pages.dev/

## Overview

NGEP is a computational framework for predicting cell-type-specific gene expression from neuronal 3D morphology. A feedforward neural network is trained on open-access data from NeuroMorpho.org and the Allen Brain Atlas to infer parvalbumin mRNA levels across regions of the mouse neocortex, using fourteen engineered morphological features derived from five base measurements.

The immediate application is Pvalb expression prediction in neocortical neurons. The broader vision is more ambitious. A framework that predicts gene expression from morphology, applied across any brain region and any gene, could map where a pathological gene is most aberrantly expressed and use that map to localize the circuit most likely responsible for a given disorder. NGEP is the first step toward that goal. Realizing that vision will require retraining on human neuronal data, as human morphology differs meaningfully from mouse, and validating that the structure-to-expression relationships learned here transfer across species.

## Clinical Significance

Parvalbumin is expressed almost exclusively in fast-spiking GABAergic interneurons, which are critical for:
- Gamma oscillations (30-100 Hz)
- Temporal precision of neural firing
- Network stability and seizure prevention
- Excitation/inhibition balance

Pvalb dysfunction is implicated in autism, schizophrenia, epilepsy, Alzheimer's disease, fragile X syndrome, and ADHD.

Measuring gene expression experimentally is expensive, slow, and often requires sacrificing tissue. A morphology-based prediction model changes that. Neuronal shape can be captured from standard microscopy, making it a far more accessible proxy for molecular state. At scale, this means expression profiling without genomic assays. Extended to disease-relevant genes across multiple brain regions, it becomes a computational route to identifying which region is contributing most to a patient's pathology. The prefrontal cortex in schizophrenia, the hippocampus in Alzheimer's, the basal ganglia in Parkinson's. Rather than measuring everything everywhere, a region-stratified expression model narrows the search to where the signal is strongest. It should be noted that mapping expression to circuit-level causation in specific disorders remains an open scientific problem; NGEP narrows the search space but does not resolve questions of causality on its own.

## Data

### Neuronal Morphology (NeuroMorpho.org)

- Source: NeuroMorpho.org REST API
- Species: Mouse
- Brain Region: Neocortex
- Sample Size: 7,500 neurons
- Format: SWC files
- Quality: Verified reconstructions from multiple labs

### Gene Expression (Allen Brain Atlas)

- Source: Allen Brain Atlas RMA API
- Gene: Parvalbumin (Pvalb)
- Method: In situ hybridization (ISH)
- Coverage: ~70 neocortical substructures, expressed as normalized expression energy

### Region Matching

Each neuron's brain region metadata is matched to Allen structures via keyword-based lookups across cortical areas, layers, and substructures, yielding region-specific expression targets with sufficient variance (~0.49) to train on.

## Morphological Features

Five base measurements are extracted from each neuron's SWC reconstruction:

1. Soma radius - Cell body size; relates to metabolism and electrical properties
2. Total dendritic length - Cumulative extent of all dendritic branches; surface area for receiving inputs
3. Bifurcation count - Number of branch points; structural complexity of the dendritic tree
4. Terminal count - Number of leaf nodes; potential synaptic endpoints
5. Branch density - Bifurcations divided by dendritic length; compactness of branching pattern

These five are expanded to fourteen engineered features through ratios, products, log transforms, and squared terms. The transformations help the network capture nonlinear interactions and better represent the compact, heavily branched morphology characteristic of parvalbumin-positive interneurons.

## Model Architecture

Neural network design:

  Input (14 features)
    -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.1)
    -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.1)
    -> Linear(1) [regression output]

Training Configuration:
  - Optimizer: Adam (lr=0.001)
  - Loss: SmoothL1Loss (robust to outliers)
  - LR schedule: ReduceLROnPlateau (factor=0.5, patience=10)
  - Batch size: 8
  - Epochs: 150
  - Cross-validation: Stratified 10-fold (stratified by brain region)
  - Random seed: 42

StandardScaler normalization is fitted independently on each training fold to prevent data leakage.

## Results

### Cross-Validation Performance

- **R²**: Mean 0.3956 (Std 0.0270) | Range 0.3400 - 0.4447
- **Pearson R**: Mean 0.6322 (Std 0.0200) | Range 0.5899 - 0.6674
- **RMSE**: Mean 0.6691 (Std 0.0155) | Range 0.6424 - 0.7012
- **MAE**: Mean 0.5073 (Std 0.0128) | Range 0.4817 - 0.5251
- **p-value**: All < 10^-51 across all 10 folds

The model explains approximately 40% of variance in Pvalb expression from morphology alone. Remaining variance is attributable to gene regulatory networks, epigenetic state, developmental history, and local circuit context. An R² standard deviation of only 0.027 across all folds indicates robust generalization. The lowest-performing fold still achieved R² = 0.34, confirming the results are not an artifact of favorable data splits. These results establish a strong proof-of-concept, though predictive accuracy would need to improve substantially before the model is suitable for direct clinical decision-making.

### Top 5 Performing Folds

- **Fold 9**: R² 0.4447 | Pearson R 0.6674 | RMSE 0.6424 | p-value 5.98e-73
- **Fold 10**: R² 0.4252 | Pearson R 0.6528 | RMSE 0.6540 | p-value 7.90e-69
- **Fold 7**: R² 0.4066 | Pearson R 0.6393 | RMSE 0.6554 | p-value 2.44e-65
- **Fold 4**: R² 0.4048 | Pearson R 0.6375 | RMSE 0.6694 | p-value 7.49e-65
- **Fold 6**: R² 0.3922 | Pearson R 0.6275 | RMSE 0.6680 | p-value 2.60e-62

### Applications

- Drug screening: Morphology-based Pvalb expression prediction from patient-derived iPSCs without requiring genomic assays
- Biomarker development: Morphological features as accessible, imaging-based Pvalb indicators
- Patient stratification: Identifying Pvalb circuit dysfunction from morphological imaging for targeted therapy
- Therapy validation: Rapid morphological confirmation of Pvalb restoration in ASD, schizophrenia, and epilepsy models
- Organoid QC: Morphological screening for healthy Pvalb interneurons prior to clinical use
- Mechanistic insight: Linking neuronal structure to molecular function to understand which morphological features drive expression

## NGEP Validator: External Validation Tool

The validator tests the trained model ensemble on external neurons from NeuroMorpho.org, providing independent confirmation that the model generalizes beyond the training set.

### Architecture

- **Frontend**: Static HTML hosted on Cloudflare Pages
- **Backend**: Flask + gunicorn hosted on Render, serving the ensemble of 10 trained fold models

### How It Works

1. Fetches fresh neurons from NeuroMorpho.org using randomized page ordering to avoid the training set
2. Filters for neocortex only, consistent with the training population
3. Extracts the same 14 engineered features using identical SWC parsing logic
4. Runs all 10 trained fold models and averages predictions
5. Compares to ground truth using a pre-computed static lookup of 129 brain region to expression energy mappings from the Allen Brain Atlas ISH data
6. Computes and returns performance metrics (R², Pearson r, RMSE, MAE)

### Avoiding Training Data Contamination

- Large population: NeuroMorpho.org has 300,000+ neurons (training set is ~2.5%)
- Random ordering: Neurons fetched from randomized API pages
- Statistical expectation: Probability of refetching a training neuron is negligible per validation sample

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

## Future Directions

  - Multi-gene prediction (GAD1, GFAP, and other disease-relevant markers)
  - Expanded brain region coverage beyond neocortex
  - Per-region stratification to resolve expression differences across cortical areas
  - Region-by-gene expression mapping across the full brain to localize circuits implicated in specific disorders
  - Human neuronal data integration to bridge the gap between mouse models and clinical application

## Acknowledgments

- NeuroMorpho.org for open-access neuron reconstructions
- Allen Brain Institute for comprehensive in situ hybridization data

Status: Stable, reproducible results across stratified 10-fold CV
Recommended for: Research, proof-of-concept, clinical biomarker development
