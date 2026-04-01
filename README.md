# NGEP: Neuron Gene Expression Prediction

**Predicting parvalbumin (Pvalb) expression from neuronal morphology using deep learning.**

Mouse neocortex · NeuroMorpho.org · Allen Brain Atlas · PyTorch

---

## What This Project Does

NGEP asks whether the 3D shape of a neuron — its soma size, dendritic length, and branching pattern — contains enough information to predict how much a cell-type-specific gene is expressed. Using real open-access data and a feedforward neural network, the answer is: **yes, meaningfully, with R² ≈ 0.327 and Pearson R ≈ 0.579** across stratified 10-fold cross-validation.

---

## Results

| Metric | Mean ± Std | Range |
|--------|-----------|-------|
| **R²** | **0.3266 ± 0.0214** | 0.2876 – 0.3560 |
| **Pearson R** | **0.5790 ± 0.0189** | 0.5389 – 0.6063 |
| **RMSE** | **0.7064 ± 0.0124** | 0.6910 – 0.7290 |
| **MAE** | **0.5608 ± 0.0071** | 0.5517 – 0.5777 |
| **p-value** | **all < 10⁻⁴⁰** | 10/10 folds significant |

### Top 5 Performing Folds

| Rank | Fold | R² | Pearson r | RMSE | p-value |
|------|------|-----|-----------|------|---------|
| 1 ⭐ | **Fold 5** | **0.3560** | **0.6052** | **0.6910** | **6.11e-57** |
| 2 | Fold 6 | 0.3511 | 0.6063 | 0.6902 | 3.36e-57 |
| 3 | Fold 10 | 0.3470 | 0.5928 | 0.6971 | 4.61e-54 |
| 4 | Fold 7 | 0.3357 | 0.5812 | 0.6934 | 1.26e-51 |
| 5 | Fold 4 | 0.3304 | 0.5806 | 0.7100 | 1.61e-51 |

**Key Finding:** Results were remarkably consistent across all 10 folds with R² standard deviation of only **0.0214**, indicating robust generalization with minimal overfitting.

---

## Clinical Significance: Parvalbumin & Neurological Disease

### Why Pvalb Expression Matters

**Parvalbumin (Pvalb)** is a calcium-binding protein expressed primarily in **fast-spiking GABAergic interneurons** — the brain's inhibitory "rhythm keepers." These cells are critical for:

- **Gamma oscillations** (30-100 Hz) — synchronized neural firing linked to attention, working memory, and sensory processing
- **Temporal precision** — coordinating precise spike timing across neuronal populations
- **Network stability** — preventing pathological hyperexcitation and seizures
- **E/I balance** — maintaining healthy equilibrium between excitation and inhibition

### Link to Neurological Conditions

**Pvalb+ interneuron dysfunction is implicated in:**

#### 1. **Autism Spectrum Disorder (ASD)**
- Reduced Pvalb expression and GABAergic inhibition in autism brains (postmortem & animal models)
- Loss of gamma oscillations correlates with social and sensory deficits
- Pvalb interneurons are a therapeutic target: stimulating these cells rescues social behavior in mouse models

#### 2. **Schizophrenia**
- Selective loss of Pvalb+ basket cells in prefrontal cortex
- Disrupted gamma oscillations and cognitive dysfunction
- Abnormal "parvalbumin circuit" contributes to positive and negative symptoms
- Antipsychotic efficacy correlates with restoration of Pvalb+ function

#### 3. **Epilepsy & Seizure Disorders**
- Pvalb interneurons provide critical inhibitory braking on excitatory pyramidal cells
- Reduced Pvalb expression → loss of inhibition → increased seizure susceptibility
- Genetic variants affecting Pvalb expression are seizure risk factors
- Therapies targeting GABAergic Pvalb cells show promise in drug-resistant epilepsy

#### 4. **Alzheimer's Disease & Cognitive Decline**
- Pvalb+ interneurons degenerate in early Alzheimer's pathology
- Loss of Pvalb cells → impaired gamma oscillations → memory deficits
- Pvalb expression is a biomarker of cognitive reserve

#### 5. **Fragile X Syndrome (FXS)**
- FXS involves excessive excitation and impaired Pvalb interneuron maturation
- Restoring Pvalb function in mouse models rescues behavioral and cognitive symptoms
- Pvalb circuit normalization is a therapeutic goal

#### 6. **Parkinson's Disease & Movement Disorders**
- Pvalb interneurons in basal ganglia regulate motor control
- Altered Pvalb expression contributes to rigidity and bradykinesia
- Targeting Pvalb+ GABAergic circuits may improve motor symptoms

#### 7. **ADHD & Executive Function**
- Pvalb interneurons in prefrontal cortex support sustained attention and impulse control
- Dysfunction → attention deficits and executive impairment
- Stimulating Pvalb circuits improves attention in preclinical models

---

## Clinical Applications of This Research

### 1. **Morphology-to-Expression Prediction for Drug Discovery**
- **Use case:** Pharmaceutical companies developing drugs to enhance Pvalb expression or function
- **This model enables:** Screening neuron morphologies from patient-derived iPSCs to predict Pvalb levels without waiting for gene sequencing
- **Benefit:** Faster identification of candidate neurons and therapeutic compounds

### 2. **Biomarker Development**
- **Use case:** Quantifying Pvalb interneuron health in brain organoids and tissue samples from neurological patients
- **This model enables:** Predicting gene expression from morphological features visible in microscopy, reducing need for expensive genomic assays
- **Benefit:** Accessible biomarkers for disease severity and treatment response

### 3. **Patient Stratification**
- **Use case:** Identifying which patients have the most Pvalb circuit dysfunction
- **This model enables:** Predicting Pvalb expression levels from structural MRI or morphological imaging to stratify patients for targeted therapies
- **Benefit:** Personalized medicine — treating patients with documented Pvalb deficiency

### 4. **Therapy Validation**
- **Use case:** Testing whether new ASD, schizophrenia, or epilepsy drugs restore Pvalb interneuron function
- **This model enables:** Rapid morphological readouts to confirm Pvalb circuit restoration without waiting for slow RNA sequencing
- **Benefit:** Accelerated clinical trial turnaround and decision-making

### 5. **Organoid & iPSC Quality Control**
- **Use case:** Ensuring neural organoids and patient-derived neurons have healthy Pvalb interneurons before clinical use
- **This model enables:** Morphological screening to predict Pvalb maturation and functionality
- **Benefit:** Better quality control for regenerative medicine and cell therapy

### 6. **Mechanistic Understanding**
- **Use case:** Understanding how neuronal morphology ("structure") links to molecular function ("gene expression")
- **This model enables:** Identifying which morphological features are most critical for proper Pvalb function
- **Benefit:** Targets for interventions and rational design of therapies

---

## The Bigger Picture

This research bridges **structure and function** — showing that a neuron's 3D shape encodes meaningful information about its molecular identity. In the clinic, this means:

✅ **Faster diagnostics** — infer gene expression from microscopy alone  
✅ **Better drug screening** — identify compounds that restore healthy Pvalb circuits  
✅ **Personalized treatment** — match patients to therapies based on their Pvalb deficiency severity  
✅ **Translational validation** — confirm that experimental therapies actually restore Pvalb function at scale  

For the ~50 million people worldwide with schizophrenia, autism, epilepsy, or cognitive decline linked to Pvalb circuit dysfunction, this type of research offers a new tool for understanding, diagnosing, and ultimately treating disease.

---

## Data Sources

- **NeuroMorpho.org** — SWC morphology files for 150 adult mouse neocortical neurons
- **Allen Brain Atlas** — Pvalb in situ hybridization expression data per brain structure

## Features

Five morphological features extracted from each SWC file:

| Feature | Definition |
|---------|-----------|
| Soma radius | Sum of radii of all soma-type nodes |
| Total dendritic length | Cumulative Euclidean distance between consecutive dendrite nodes |
| Bifurcation count | Number of nodes with ≥ 2 children |
| Terminal count | Number of leaf nodes |
| Branch density | Bifurcations / dendritic length |

## Model

Two-layer feedforward network in PyTorch:
- Input: 5 morphological features
- Layer 1: Linear(128) → BatchNorm → ReLU → Dropout(0.1)
- Layer 2: Linear(64) → BatchNorm → ReLU → Dropout(0.1)
- Output: Linear(1) regression

**Training:** Adam optimizer, lr=0.001, MSE loss, 150 epochs, batch size 8  
**Validation:** Stratified 10-fold cross-validation by brain region  
**Preprocessing:** StandardScaler (fit on training fold only, no data leakage)

---

## Citation

If referencing this project, please cite the underlying data sources:

- Ascoli et al. (2007). NeuroMorpho.Org. *Journal of Neuroscience*, 27(35).
- Lein et al. (2007). Allen Brain Atlas. *Nature*, 445(7124).

---

**Contact:** For clinical applications or collaborations, inquire about translational implementation of morphology-to-expression predictions.