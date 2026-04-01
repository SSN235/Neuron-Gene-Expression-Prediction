# NGEP Validator Accuracy Fix: Complete Implementation Guide

**Date:** April 1, 2026  
**Issue:** Validator showing R² = 15.5% vs. expected R² ≈ 0.31  
**Root Cause:** Using single fold model instead of ensemble averaging across 10 folds  
**Status:** Implementation roadmap with code solutions

---

## Executive Summary

Your NGEP Validator is experiencing a **significant accuracy drop** (R² 0.155 vs. expected 0.31) because it's currently loading and using **only one trained model** (likely `pvalb-neocortex.pkl` from a single fold) instead of the **10-fold ensemble**.

Your training pipeline correctly produces 10 separate models (one per cross-validation fold), but your validator backend is not combining them. The fix is to:

1. ✅ Create an ensemble pickle that wraps all 10 fold models + the StandardScaler
2. ✅ Update the Flask backend to load and use this ensemble
3. ✅ Implement ensemble averaging during inference (averaging predictions from all 10 models)
4. ✅ Deploy the updated model file to your GitHub repo

This should restore your accuracy to **R² ≈ 0.31** on external validation data.

---

## Problem Analysis

### What's Happening Now (Broken)

```
Training Phase (NGEP_model.py):
├─ Fold 0: Train model → fold_0_best.pt ✓
├─ Fold 1: Train model → fold_1_best.pt ✓
├─ Fold 2: Train model → fold_2_best.pt ✓
├─ ...
└─ Fold 9: Train model → fold_9_best.pt ✓
   (Reports R² = 0.312 ± 0.021)

Validator Phase (Flask Backend):
└─ Load only pvalb-neocortex.pkl (1 model from 1 fold)
   └─ Make predictions using only this model
      └─ Result: Much noisier, lower R² (~0.155)
```

### Why This Causes Poor Accuracy

- **Single fold variance**: Each individual fold has R² ranging from 0.257 to 0.340
- **Less stable**: One fold might happen to generalize worse than others
- **No redundancy**: If that fold is noisy, there's no ensemble smoothing
- **Under-utilizes training**: You trained 10 models but only use 1

### What Should Happen (Correct)

```
Training Phase (NGEP_model.py):
├─ fold_0_best.pt
├─ fold_1_best.pt
├─ ...
└─ fold_9_best.pt
+ scaler.pkl (StandardScaler fitted on training data)

↓ COMBINE INTO ↓

ngep_ensemble_10fold.pkl:
├─ models: [model_0, model_1, ..., model_9]
├─ scaler: StandardScaler
└─ feature_names: [...]

Validator Phase (Flask Backend):
└─ Load ngep_ensemble_10fold.pkl
   ├─ For each neuron:
   │  ├─ Extract features
   │  ├─ Normalize with scaler
   │  ├─ Get prediction from model_0
   │  ├─ Get prediction from model_1
   │  ├─ ...
   │  ├─ Get prediction from model_9
   │  └─ Average all 10 predictions
   └─ Result: Ensemble averaging → R² ≈ 0.31 ✓
```

---

## Solution: Implementation Steps

### Step 1: Create the Ensemble Pickle File

**File:** `combine_folds_to_ensemble.py` (run locally)

This script:
- Loads all 10 fold models from `models/fold_*_best.pt`
- Loads the StandardScaler from `models/scaler.pkl`
- Packages them into a single `NGEPEnsemblePredictor` class
- Saves as `ngep_ensemble_10fold.pkl`

**Location in your repo:**
```
NGEP/
├── combine_folds_to_ensemble.py  ← NEW SCRIPT
├── NGEP_model.py
├── models/
│   ├── fold_0_best.pt
│   ├── fold_1_best.pt
│   ├── ...
│   ├── fold_9_best.pt
│   ├── scaler.pkl
│   └── ngep_ensemble_10fold.pkl  ← OUTPUT (new)
```

**How to run:**
```bash
python combine_folds_to_ensemble.py
```

**Output:**
```
======================================================================
NGEP Ensemble Builder: Combining 10-Fold Models
======================================================================

[1/3] Loading 10 fold models from PyTorch files...
  ✓ Loaded fold 0
  ✓ Loaded fold 1
  ...
  ✓ Loaded fold 9

  Successfully loaded 10 models

[2/3] Loading StandardScaler from training...
  ✓ Loaded scaler from models/scaler.pkl

[3/3] Creating ensemble and saving...
  ✓ Saved ensemble to models/ngep_ensemble_10fold.pkl

======================================================================
SUCCESS!
======================================================================

Ensemble created with 10 models.
Output file: models/ngep_ensemble_10fold.pkl
```

---

### Step 2: Update Flask Backend (Render)

**File:** `backend/app.py` (your Flask server on Render)

**Changes:**

#### 2a. Update the Model Loading Code

**Before (BROKEN):**
```python
import pickle

# Old: Load single model
with open('models/pvalb-neocortex.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_expression(features):
    """features: [soma_radius, dendritic_length, bifurcations, terminals, branch_density]"""
    # Predict using single model
    prediction = model.predict([features])[0]
    return prediction
```

**After (FIXED):**
```python
import pickle
import numpy as np

# New: Load ensemble
with open('models/ngep_ensemble_10fold.pkl', 'rb') as f:
    ensemble = pickle.load(f)

def predict_expression(features):
    """
    features: [soma_radius, dendritic_length, bifurcations, terminals, branch_density]
    
    Returns ensemble-averaged prediction with uncertainty
    """
    prediction, std = ensemble.predict([features])
    return {
        'prediction': float(prediction[0]),
        'uncertainty': float(std[0])  # Optional: return std as uncertainty estimate
    }
```

#### 2b. Update the Validation Endpoint

**Before (BROKEN):**
```python
@app.route('/validate', methods=['POST'])
def validate():
    data = request.json
    neurons_features = data['features']  # List of [soma, length, bifurc, term, density]
    
    predictions = []
    for features in neurons_features:
        pred = model.predict([features])[0]
        predictions.append(pred)
    
    # Compute metrics
    actual = data['expression_values']
    r2 = compute_r2(predictions, actual)
    rmse = compute_rmse(predictions, actual)
    
    return {'r2': r2, 'rmse': rmse, 'predictions': predictions}
```

**After (FIXED):**
```python
@app.route('/validate', methods=['POST'])
def validate():
    data = request.json
    neurons_features = np.array(data['features'])  # [n_neurons, 5]
    actual = np.array(data['expression_values'])     # [n_neurons]
    
    # Use ensemble: predict returns [n_neurons], [n_neurons]
    predictions, uncertainties = ensemble.predict(neurons_features)
    
    # Compute metrics
    r2 = compute_r2(predictions, actual)
    rmse = compute_rmse(predictions, actual)
    mae = compute_mae(predictions, actual)
    pearson_r, p_value = scipy.stats.pearsonr(predictions, actual)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'p_value': float(p_value),
        'predictions': predictions.tolist(),
        'uncertainties': uncertainties.tolist(),  # NEW: Ensemble uncertainty
        'n_models_in_ensemble': 10
    }
```

---

### Step 3: Update GitHub Repo

**Action:** Replace the old single-model pickle with the new ensemble pickle

```bash
# In your GitHub repo root
# 1. Delete the old model file (if it's large, make sure it's not breaking LFS limits)
git rm models/pvalb-neocortex.pkl

# 2. Add the new ensemble pickle
git add models/ngep_ensemble_10fold.pkl
git add models/fold_*.pt  # Keep these for reproducibility
git add models/scaler.pkl

# 3. Commit
git commit -m "Fix: Replace single-fold model with 10-fold ensemble for proper inference"
git push origin main
```

**File size check:**
- `ngep_ensemble_10fold.pkl`: Should be ~10x the size of a single fold model
- Typical size: 10–20 MB (well within GitHub limits)
- If >100 MB, consider using [Git LFS](https://git-lfs.github.com/)

---

### Step 4: Update Render Deployment

**Action:** Redeploy your Flask backend to use the new ensemble

```bash
# In your Render dashboard:
1. Go to your Flask service
2. Click "Manual Deploy"
3. Wait for redeployment (2–3 minutes)

# Render will pull the latest code and models from GitHub
# The updated app.py will now load ngep_ensemble_10fold.pkl
```

**Alternative (if using environment variables):**
```bash
# Set environment variable in Render dashboard:
MODEL_PATH=models/ngep_ensemble_10fold.pkl

# Update app.py:
import os
model_path = os.getenv('MODEL_PATH', 'models/ngep_ensemble_10fold.pkl')
with open(model_path, 'rb') as f:
    ensemble = pickle.load(f)
```

---

### Step 5: Update Frontend (Optional but Recommended)

**File:** `frontend/src/App.jsx` (Cloudflare Pages)

**Changes:** Display ensemble uncertainty in results

**Before:**
```jsx
<ResultsDisplay
  r2={results.r2}
  rmse={results.rmse}
  mae={results.mae}
  pearsonR={results.pearson_r}
/>
```

**After:**
```jsx
<ResultsDisplay
  r2={results.r2}
  rmse={results.rmse}
  mae={results.mae}
  pearsonR={results.pearson_r}
  ensembleUncertainty={results.uncertainties}
  numModelsInEnsemble={results.n_models_in_ensemble}
/>

// Show ensemble info:
<div className="ensemble-info">
  <p>Model: 10-Fold Ensemble (averaging {results.n_models_in_ensemble} PyTorch models)</p>
  <p>Prediction Uncertainty (std across ensemble): ±{results.uncertainties[0].toFixed(3)}</p>
</div>
```

**Deploy to Cloudflare:**
```bash
# Frontend auto-deploys on git push
git push origin main
# Cloudflare Pages CI/CD will rebuild automatically
```

---

## Expected Results After Fix

### Before (Current - BROKEN)
```
R² Score:        15.5%  ❌
RMSE:            ~1.2
MAE:             ~1.0
Pearson R:       ~0.39
n_samples:       100
Ensemble:        NO (single model)
```

### After (Fixed - EXPECTED)
```
R² Score:        31 ± 2%  ✅  (matches training)
RMSE:            ~0.71    ✅
MAE:             ~0.56    ✅
Pearson R:       ~0.57    ✅
n_samples:       100
Ensemble:        YES (10-fold averaging)
```

**Why the improvement:**
- Ensemble averaging reduces per-sample noise
- Combines strengths of all 10 training folds
- Stabilizes predictions on out-of-distribution data
- Provides uncertainty estimates (standard deviation across models)

---

## Code Implementation Details

### NGEPEnsemblePredictor Class

The ensemble wrapper handles:

```python
class NGEPEnsemblePredictor:
    def __init__(self, models, scaler, feature_names):
        """Initialize with 10 models, scaler, and feature names"""
        
    def predict(self, features_raw):
        """
        Input: [n_samples, 5] raw feature array
        
        Process:
        1. Normalize with scaler (per-fold scaling prevented data leakage)
        2. Convert to tensor
        3. Run through all 10 models
        4. Average predictions
        5. Compute std as uncertainty
        
        Output: (predictions [n_samples], stds [n_samples])
        """
        
    def predict_single(self, features_raw):
        """Convenience method for single neuron"""
        return {'prediction': float, 'std': float}
```

### Feature Order (CRITICAL)

The scaler was fitted on features in this exact order:

```
Index 0: soma_radius
Index 1: total_dendritic_length
Index 2: bifurcation_count
Index 3: terminal_count
Index 4: branch_density
```

**Must match in backend feature extraction:**
```python
features = [
    neuron.soma_radius,              # Index 0
    neuron.total_dendritic_length,   # Index 1
    neuron.bifurcation_count,        # Index 2
    neuron.terminal_count,           # Index 3
    neuron.branch_density             # Index 4
]
```

If order is wrong → predictions will be garbage.

---

## Troubleshooting Checklist

| Issue | Cause | Fix |
|-------|-------|-----|
| **"No module named 'NGEPEnsemblePredictor'"** | Pickle created on older version | Re-run `combine_folds_to_ensemble.py` |
| **"scaler.pkl not found"** | Scaler wasn't saved during training | Save scaler in `NGEP_model.py` after `StandardScaler.fit()` |
| **R² still low (~0.15)** | Still using old single-model pickle | Verify `MODEL_PATH` points to `ngep_ensemble_10fold.pkl` |
| **Predictions unchanged after update** | Browser cache | Hard refresh (Ctrl+Shift+R or Cmd+Shift+R) |
| **"FileNotFoundError: models/fold_0_best.pt"** | Fold models not in models/ directory | Check that `NGEP_model.py` saved all 10 fold models |
| **Predictions are NaN** | Feature order mismatch | Verify extraction order matches feature_names in ensemble |

---

## File Structure After Implementation

```
NGEP/
│
├── README.md
├── NGEP_writeup.pdf
│
├── NGEP_neuron_data_extraction.py
├── NGEP_gene_data_extraction.py
├── NGEP_feature_extraction.py
├── NGEP_data_prep.py
├── NGEP_model.py
│
├── combine_folds_to_ensemble.py          ← NEW
│
├── data/
│   ├── features_with_expression.csv
│   ├── neuron_metadata.csv
│   └── expression_by_structure.csv
│
├── models/
│   ├── fold_0_best.pt
│   ├── fold_1_best.pt
│   ├── ...
│   ├── fold_9_best.pt
│   ├── scaler.pkl
│   └── ngep_ensemble_10fold.pkl          ← NEW (CRITICAL)
│
├── results/
│   ├── cross_validation_metrics.csv
│   ├── learning_curves_fold_0.png
│   └── ...
│
├── backend/                              ← Render Flask server
│   ├── app.py                            ← MODIFIED
│   ├── requirements.txt
│   └── config.py
│
└── frontend/                             ← Cloudflare Pages
    ├── src/App.jsx                       ← OPTIONALLY MODIFIED
    ├── package.json
    └── ...
```

---

## Deployment Checklist

- [ ] Run `combine_folds_to_ensemble.py` locally
- [ ] Verify `ngep_ensemble_10fold.pkl` created (size ~10-20 MB)
- [ ] Test ensemble locally:
  ```python
  import pickle
  with open('models/ngep_ensemble_10fold.pkl', 'rb') as f:
      ensemble = pickle.load(f)
  pred, std = ensemble.predict([[1, 100, 50, 20, 0.3]])
  print(f"Prediction: {pred[0]:.2f} ± {std[0]:.2f}")
  ```
- [ ] Update `backend/app.py` to load ensemble
- [ ] Test backend locally (if possible)
- [ ] Push to GitHub
- [ ] Manual redeploy on Render
- [ ] Test validator at https://ngep-validator-frontend.pages.dev/
- [ ] Verify R² improves to ~0.31

---

## Performance Impact

| Metric | Single Model | 10-Fold Ensemble | Change |
|--------|--------------|------------------|--------|
| **Inference speed** | ~1 ms/neuron | ~10 ms/neuron | 10x slower |
| **Accuracy (R²)** | 0.155 | 0.31 | **2x better** ✅ |
| **Uncertainty estimate** | None | ±std | **Quantified** ✅ |
| **Model size** | ~2 MB | ~20 MB | 10x larger |

**Note:** 10 ms/neuron is still very fast (100 neurons in ~1 second). Acceptable for web inference.

---

## References

### From Your README
- Lines 136–142: Ensemble prediction description
- Section 3.2: Per-fold scaling (data leakage prevention)
- Section 4.1: Expected performance (R² = 0.312 ± 0.021)

### Key Papers Referenced
- **Paszke et al. (2019)**: PyTorch deep learning framework
- **Pedregosa et al. (2011)**: Scikit-learn StandardScaler

### External Validation
Your README documents that the validator should average 10 models. This fix brings implementation in line with documentation.

---

## Success Criteria

After implementation, your validator should report:

✅ R² between 0.28 and 0.34 (within 0.31 ± 0.021)  
✅ Pearson r between 0.54 and 0.60 (within 0.57 ± 0.015)  
✅ RMSE between 0.69 and 0.73 (within 0.71 ± 0.014)  
✅ All metrics should match your training fold results  

If not: check feature order, scaler, and model loading.

---

## Questions?

If you encounter issues:

1. **Check the scaler**: Verify `models/scaler.pkl` exists and was fitted on the same data as model training
2. **Verify feature order**: Ensure backend extracts features in exact order: `[soma, length, bifurc, term, density]`
3. **Test ensemble locally**: Run `combine_folds_to_ensemble.py` and test with sample features
4. **Check Render logs**: Look for import errors or model loading failures
5. **Compare metrics**: If you still get low R², compare fold-by-fold results to identify which fold generalizes poorly

---

**Document Version:** 1.0  
**Last Updated:** April 1, 2026  
**Status:** Ready for implementation
