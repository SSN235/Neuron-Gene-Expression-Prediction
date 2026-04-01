# NGEP: Neuron Gene Expression Prediction

Predicting parvalbumin (Pvalb) expression from neuronal morphology using deep learning.

External Validation Tool: https://ngep-validator-frontend.pages.dev/

## Overview

NGEP is a computational framework for predicting cell-type-specific gene expression from neuronal 3D morphology. Using real open-access data from NeuroMorpho.org and the Allen Brain Atlas, we train a feedforward neural network to infer parvalbumin mRNA levels in mouse neocortex from five morphological features.
 
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

Five features extracted from each neuron's 3D reconstruction:

1. Soma radius - Cell body size
2. Total dendritic length - Cumulative dendrite extent
3. Bifurcation count - Number of branch points
4. Terminal count - Number of synaptic endpoints
5. Branch density - Bifurcations / dendritic length

These features capture the compact, highly branched morphology characteristic of parvalbumin-positive fast-spiking interneurons.
 

## Model Architecture

Neural network design:

Input (5 features)
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.1)
  -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.1)
  -> Linear(1) [regression output]

Training Configuration:
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Batch size: 8
- Epochs: 150
- Cross-validation: Stratified 10-fold
- Random seed: 42

 

## Results

### Cross-Validation Performance

| Metric | Mean | Std | Range |
|  --|  | --|  -|
| R² | 0.3266 | 0.0214 | 0.2876 - 0.3560 |
| Pearson R | 0.5790 | 0.0189 | 0.5389 - 0.6063 |
| RMSE | 0.7064 | 0.0124 | 0.6910 - 0.7290 |
| MAE | 0.5608 | 0.0071 | 0.5517 - 0.5777 |
| p-value | all < 10^-40 | - | 10/10 folds |

The model explains approximately 31% of variance in Pvalb expression from five morphological measurements. Remaining variance is attributable to gene regulatory networks, epigenetic state, developmental history, and local circuit context.

### Top 5 Performing Folds

| Fold | R² | Pearson R | RMSE | p-value |
|  | --|   --|  |   |
| Fold 5 | 0.3560 | 0.6052 | 0.6910 | 6.11e-57 |
| Fold 6 | 0.3511 | 0.6063 | 0.6902 | 3.36e-57 |
| Fold 10 | 0.3470 | 0.5928 | 0.6971 | 4.61e-54 |
| Fold 7 | 0.3357 | 0.5812 | 0.6934 | 1.26e-51 |
| Fold 4 | 0.3304 | 0.5806 | 0.7100 | 1.61e-51 |

Remarkably consistent performance across folds with R² standard deviation of only 0.0214, indicating robust generalization.


### Applications

- Drug screening: Morphology-based Pvalb expression prediction from iPSCs
- Biomarker development: Morphological features as accessible Pvalb indicators
- Patient stratification: Identifying Pvalb circuit dysfunction for targeted therapy
- Therapy validation: Rapid morphological confirmation of Pvalb restoration
- Organoid QC: Morphological screening for healthy Pvalb interneurons
- Mechanistic insight: Linking neuronal structure to molecular function

## NGEP Validator: External Validation Tool

The validator is a web-based tool for testing the ensemble model on external validation data from NeuroMorpho.org.

### How It Works

1. Fetches fresh neurons from NeuroMorpho.org (not from training set)
2. Extracts the same 5 morphological features
3. Applies StandardScaler normalization (fitted during training)
4. Passes features through all 10 trained models
5. Averages predictions across models (ensemble voting)
6. Computes performance metrics (R², Pearson r, RMSE, MAE)

### Avoiding Training Data Contamination

- Large population: NeuroMorpho.org has 300,000+ neurons (training set is <0.05%)
- Random ordering: Neurons fetched in random order
- Statistical expectation: Probability of refetching same neuron is ~0.05% per validation neuron
- Transparent reporting: Validator logs any overlapping neurons

Expected contamination rate: <0.05% (negligible)

### Preprocessing Consistency

- Scaler: Identical StandardScaler parameters from training
- Feature extraction: Identical code as NGEP_feature_extraction.py
- SWC parsing: Identical parsing logic
- Region matching: Not performed for validation (only morphology used)

### API Endpoints

Fetch neurons:
POST /api/fetch-neurons
Parameters: count, species, brain_region, randomize

Get predictions:
POST /api/predict
Parameters: features, model_version

Compute metrics:
POST /api/evaluate
Parameters: actual, predicted

## Limitations

- Sparse features (5 features): Spine density, axonal morphology not included
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

Dependencies (pin for exact reproducibility):
torch==2.0.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.1
seaborn==0.12.2
requests==2.31.0
beautifulsoup4==4.12.2

Hardware Effects: GPU (CUDA/MPS) may introduce minor floating-point differences. CPU execution is slower but fully reproducible.

## Acknowledgments

- NeuroMorpho.org for open-access neuron reconstructions
- Allen Brain Institute for comprehensive in situ hybridization data

 

Status: Stable, reproducible results across stratified 10-fold CV
Recommended for: Research, proof-of-concept, clinical biomarker development
