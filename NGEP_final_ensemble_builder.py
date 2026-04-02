"""
NGEP Ensemble Builder - 14 Feature Version
============================================

Combines all 10 fold models into a single ensemble pickle.
Matches NGEP_model.py which trains on 14 engineered features.

IMPORTANT: Run NGEP_model.py first to generate the fold_X_best.pt files,
then run this script to package them into a single deployable pickle.
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")


# ──────────────────────────────────────────────────────────────────────
# Feature engineering (must match NGEP_model.py exactly)
# ──────────────────────────────────────────────────────────────────────

BASE_FEATURE_NAMES = [
    'soma_radius', 'total_dendritic_length', 'bifurcations',
    'terminals', 'branch_density',
]

ALL_FEATURE_NAMES = [
    'soma_radius', 'total_dendritic_length', 'bifurcations', 'terminals',
    'branch_density', 'ratio_dend_soma', 'ratio_bifur_term',
    'ratio_term_dend', 'product_soma_dend', 'product_bifur_term',
    'log_bifurcations', 'log_terminals', 'soma_radius_sq', 'dend_length_sq',
]

NUM_FEATURES = 14


def engineer_features(X_base: np.ndarray) -> np.ndarray:
    """
    Engineer 14 features from 5 base features.
    Must match NGEP_model.py lines 57-76 exactly.
    """
    return np.column_stack([
        X_base,
        # Ratios
        X_base[:, 1] / (X_base[:, 0] + 1e-6),  # dendritic/soma
        X_base[:, 2] / (X_base[:, 3] + 1e-6),  # bifurcations/terminals
        X_base[:, 3] / (X_base[:, 1] + 1e-6),  # terminals/dendritic
        # Products
        X_base[:, 0] * X_base[:, 1],            # soma*dendritic
        X_base[:, 2] * X_base[:, 3],            # bifurcations*terminals
        # Log transforms
        np.log(X_base[:, 2] + 1),               # log bifurcations
        np.log(X_base[:, 3] + 1),               # log terminals
        # Squares
        X_base[:, 0] ** 2,                      # soma²
        X_base[:, 1] ** 2,                      # dendritic²
    ])


# ──────────────────────────────────────────────────────────────────────
# Model definition (must match NGEP_model.py NeuralNetwork class)
# ──────────────────────────────────────────────────────────────────────

class NGEPModel(nn.Module):
    """Neural network model matching NGEP_model.py NeuralNetwork class."""

    def __init__(self, input_size=14, hidden1=128, hidden2=64,
                 dropout=0.1, use_batch_norm=False):
        super(NGEPModel, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.fc1 = nn.Linear(input_size, hidden1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Alias so pickle can resolve either name
NeuralNetwork = NGEPModel


class NGEPEnsemblePredictor:
    """10-fold ensemble predictor with 14 engineered features."""

    def __init__(self, models: List[NGEPModel], scaler, feature_names: List[str]):
        self.models = models
        self.scaler = scaler
        self.feature_names = feature_names
        self.num_features = NUM_FEATURES
        self.num_models = len(models)

        if self.num_models != 10:
            raise ValueError(f"Expected 10 models, got {self.num_models}")

        for model in self.models:
            model.eval()

    def predict(self, features_engineered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble averaging.

        Input: already-engineered features with shape (n, 14).
        The scaler was fit on 14 engineered features during training.
        """
        features_engineered = np.array(features_engineered).reshape(-1, self.num_features)
        features_normalized = self.scaler.transform(features_engineered)
        features_tensor = torch.FloatTensor(features_normalized)

        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(features_tensor).numpy().flatten()
                all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        predictions = np.mean(all_predictions, axis=0)
        stds = np.std(all_predictions, axis=0)

        return predictions, stds

    def predict_single(self, features_engineered: List[float]) -> dict:
        pred, std = self.predict([features_engineered])
        return {'prediction': float(pred[0]), 'std': float(std[0])}

    def __repr__(self):
        return f"NGEPEnsemblePredictor(models={self.num_models}, features={self.num_features})"


# ──────────────────────────────────────────────────────────────────────
# Build ensemble
# ──────────────────────────────────────────────────────────────────────

def combine_folds_to_ensemble(models_dir='models',
                              output_file='models/ngep_ensemble_14feat.pkl',
                              data_file='data/features_with_expression.csv'):
    """Load 10 fold models, generate scaler on 14 features, package ensemble."""

    models_dir = Path(models_dir)
    output_file = Path(output_file)

    print("=" * 70)
    print("NGEP Ensemble Builder (14 Engineered Features)")
    print("=" * 70)

    # ── Step 1: Detect architecture ──────────────────────────────────
    print("\n[1/4] Detecting architecture...")
    first_model_path = models_dir / 'fold_0_best.pt'
    if not first_model_path.exists():
        raise FileNotFoundError(f"Model not found: {first_model_path}")

    state_dict = torch.load(str(first_model_path), map_location='cpu')
    has_batch_norm = 'bn1.weight' in state_dict or 'bn2.weight' in state_dict

    # Check input size from fc1 weight shape
    fc1_in = state_dict['fc1.weight'].shape[1]
    print(f"  Architecture: input_size={fc1_in}, "
          f"{'with' if has_batch_norm else 'without'} BatchNorm")

    if fc1_in != NUM_FEATURES:
        print(f"\n  WARNING: Model was trained with {fc1_in} features "
              f"but expected {NUM_FEATURES}.")
        print(f"  Make sure you re-ran NGEP_model.py with feature engineering.")
        if fc1_in == 5:
            print(f"  The .pt files appear to be from the OLD 5-feature training.")
            print(f"  Please re-run NGEP_model.py first, then run this script again.")
            raise ValueError(f"Model input_size={fc1_in}, expected {NUM_FEATURES}")

    # ── Step 2: Load 10 fold models ──────────────────────────────────
    print(f"\n[2/4] Loading 10 fold models (input_size={fc1_in})...")
    models = []

    for fold_idx in range(10):
        model_path = models_dir / f'fold_{fold_idx}_best.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = NGEPModel(input_size=fc1_in, hidden1=128, hidden2=64,
                          dropout=0.1, use_batch_norm=has_batch_norm)
        sd = torch.load(str(model_path), map_location='cpu')
        model.load_state_dict(sd)
        models.append(model)
        print(f"  Fold {fold_idx}: OK")

    # ── Step 3: Generate scaler on 14 engineered features ────────────
    print(f"\n[3/4] Generating scaler from training data...")
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=BASE_FEATURE_NAMES + ['expression_energy'])

    X_base = df[BASE_FEATURE_NAMES].values
    X_engineered = engineer_features(X_base)

    print(f"  Base features shape: {X_base.shape}")
    print(f"  Engineered features shape: {X_engineered.shape}")

    scaler = StandardScaler()
    scaler.fit(X_engineered)
    print(f"  Scaler fit on {X_engineered.shape[0]} samples, {X_engineered.shape[1]} features")

    # Also save standalone scaler
    scaler_path = models_dir / 'scaler_14feat.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")

    # ── Step 4: Create and save ensemble ─────────────────────────────
    print(f"\n[4/4] Creating ensemble...")
    ensemble = NGEPEnsemblePredictor(
        models=models,
        scaler=scaler,
        feature_names=ALL_FEATURE_NAMES,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(ensemble, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_file.stat().st_size / (1024 * 1024)

    # ── Quick validation ─────────────────────────────────────────────
    print(f"\n  Quick validation test...")
    test_features = X_engineered[:5]
    preds, stds = ensemble.predict(test_features)
    print(f"  Test predictions: {preds}")
    print(f"  Test stds: {stds}")

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"  Ensemble: {output_file} ({size_mb:.1f} MB)")
    print(f"  Models: {ensemble.num_models}")
    print(f"  Features: {ensemble.num_features}")
    print(f"  Feature names: {ALL_FEATURE_NAMES}")
    print(f"\n  Deploy this file to your backend models/ folder.")
    print("=" * 70)

    return ensemble


if __name__ == '__main__':
    try:
        ensemble = combine_folds_to_ensemble()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)