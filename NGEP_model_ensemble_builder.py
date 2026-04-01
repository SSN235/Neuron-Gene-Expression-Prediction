"""
NGEP Ensemble Builder - Simplified
===================================

Combines all 10 fold models into a single ensemble pickle.
Auto-detects model architecture and generates scaler if missing.
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

class NGEPModel(nn.Module):
    """Neural network model - flexible to handle with/without BatchNorm."""
    
    def __init__(self, input_size=5, hidden1=128, hidden2=64, dropout=0.1, use_batch_norm=True):
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


class NGEPEnsemblePredictor:
    """10-fold ensemble predictor with uncertainty estimates."""
    
    def __init__(self, models: List[NGEPModel], scaler, feature_names: List[str]):
        self.models = models
        self.scaler = scaler
        self.feature_names = feature_names
        self.num_models = len(models)
        
        if self.num_models != 10:
            raise ValueError(f"Expected 10 models, got {self.num_models}")
        
        for model in self.models:
            model.eval()
    
    def predict(self, features_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble averaging."""
        features_raw = np.array(features_raw).reshape(-1, 5)
        features_normalized = self.scaler.transform(features_raw)
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
    
    def predict_single(self, features_raw: List[float]) -> dict:
        """Single neuron prediction."""
        pred, std = self.predict([features_raw])
        return {'prediction': float(pred[0]), 'std': float(std[0])}
    
    def __repr__(self):
        return f"NGEPEnsemblePredictor(models={self.num_models})"


def generate_scaler_from_data(data_file: str = 'data/features_with_expression.csv',
                              scaler_output: str = 'models/scaler.pkl',
                              feature_columns: list = None):
    """Generate scaler from training data if missing."""
    
    data_file = Path(data_file)
    
    if feature_columns is None:
        feature_columns = ['soma_radius', 'total_dendritic_length', 'bifurcations',
                          'terminals', 'branch_density']
    
    print("  Generating scaler from training data...")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    if not all(col in df.columns for col in feature_columns):
        missing = [col for col in feature_columns if col not in df.columns]
        raise ValueError(f"CSV missing columns: {missing}")
    
    features = df[feature_columns].values
    scaler = StandardScaler()
    scaler.fit(features)
    
    Path(scaler_output).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_output, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"  ✓ Scaler saved")
    return scaler


def combine_folds_to_ensemble(models_dir: str = 'models',
                              output_file: str = 'models/ngep_ensemble_10fold.pkl',
                              data_file: str = 'data/features_with_expression.csv') -> NGEPEnsemblePredictor:
    """Load and combine all 10 fold models into ensemble."""
    
    models_dir = Path(models_dir)
    output_file = Path(output_file)
    
    feature_names = ['soma_radius', 'total_dendritic_length', 'bifurcations',
                    'terminals', 'branch_density']
    
    print("="*70)
    print("NGEP Ensemble Builder")
    print("="*70)
    
    # Detect architecture
    print("\n[1/3] Detecting architecture...")
    first_model_path = models_dir / 'fold_0_best.pt'
    if not first_model_path.exists():
        raise FileNotFoundError(f"Model not found: {first_model_path}")
    
    state_dict = torch.load(str(first_model_path), map_location='cpu')
    has_batch_norm = 'bn1.weight' in state_dict or 'bn2.weight' in state_dict
    print(f"  Architecture: {'with' if has_batch_norm else 'without'} BatchNorm")
    
    # Load models
    print("\n[2/3] Loading 10 fold models...")
    models = []
    
    for fold_idx in range(10):
        model_path = models_dir / f'fold_{fold_idx}_best.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            model = NGEPModel(input_size=5, hidden1=128, hidden2=64, dropout=0.1,
                            use_batch_norm=has_batch_norm)
            state_dict = torch.load(str(model_path), map_location='cpu')
            model.load_state_dict(state_dict)
            models.append(model)
            print(f"  ✓ Fold {fold_idx}")
        
        except RuntimeError as e:
            print(f"\nError loading fold {fold_idx}: {e}")
            print("Run: python inspect_fold_model.py")
            raise
    
    # Load or generate scaler
    print("\n[3/3] Loading scaler...")
    scaler_path = models_dir / 'scaler.pkl'
    
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  ✓ Scaler loaded")
    else:
        scaler = generate_scaler_from_data(data_file, str(scaler_path), feature_names)
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble = NGEPEnsemblePredictor(models=models, scaler=scaler, feature_names=feature_names)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(ensemble, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print("="*70)
    print(f"SUCCESS")
    print("="*70)
    print(f"Ensemble saved: {output_file} ({size_mb:.1f} MB)")
    print(f"Ready for deployment!")
    print("="*70)
    
    return ensemble


if __name__ == '__main__':
    try:
        ensemble = combine_folds_to_ensemble()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)