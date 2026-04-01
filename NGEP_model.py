 #imported libraries necessary for training regression model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

# ===== CONFIGURATION =====
# User can easily modify these settings
DATA_FILE = 'data/features_with_expression.csv'
USE_GPU = True  # Set to False to use CPU only
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
HIDDEN_1 = 128  # Number of units in first hidden layer
HIDDEN_2 = 64  # Number of units in second hidden layer
DROPOUT = 0.1  # Dropout rate

NUM_FOLDS = 10  # Number of folds for cross-validation
RANDOM_STATE = 42
# =====================

print("Starting data loading...")

#loads data from CSV file
data = pd.read_csv(DATA_FILE)
print(f"Loaded data. Shape: {data.shape}")

#defines feature columns and target column
feature_cols = ['soma_radius', 'total_dendritic_length', 'bifurcations', 'terminals', 'branch_density']
target_col = 'expression_energy'

#removes rows with missing values in features or target
initial_rows = len(data)
data = data.dropna(subset=feature_cols + [target_col])
removed_rows = initial_rows - len(data)
if removed_rows > 0:
    print(f"Removed {removed_rows} rows with missing values")

#extracts X (features) and y (target) as numpy arrays
# NEW - Feature Engineering
X_base = data[feature_cols].values

X = np.column_stack([
    X_base,
    
    # ===== RATIOS (capture morphological relationships) =====
    X_base[:, 1] / (X_base[:, 0] + 1e-6),      # dendritic/soma: neurons with long dendrites relative to soma size may express differently
    X_base[:, 2] / (X_base[:, 3] + 1e-6),      # bifurcations/terminals: branching efficiency (more branches per endpoint = different structure)
    X_base[:, 3] / (X_base[:, 1] + 1e-6),      # terminals/dendritic: terminal density (how densely packed are endpoints along the dendrite)
    
    # ===== PRODUCTS (capture complexity and interactions) =====
    X_base[:, 0] * X_base[:, 1],               # soma*dendritic: overall neuron size (scale factor combining two dimensions)
    X_base[:, 2] * X_base[:, 3],               # bifurcations*terminals: branching complexity (highly branched neurons with many endpoints are fundamentally different)
    
    # ===== LOG TRANSFORMS (normalize skewed distributions) =====
    np.log(X_base[:, 2] + 1),                  # log bifurcations: bifurcation counts are usually right-skewed (few neurons have extremely high counts); log makes distribution normal
    np.log(X_base[:, 3] + 1),                  # log terminals: same as bifurcations; log helps model learn exponential relationships better
    
    # ===== SQUARES (capture nonlinear effects) =====
    X_base[:, 0] ** 2,                         # soma²: surface area and volume scale nonlinearly with radius; squared term lets model learn this without explicit formula
    X_base[:, 1] ** 2,                         # dendritic²: same reasoning; nonlinear growth effects that impact expression
])


feature_cols = ['soma_radius', 'total_dendritic_length', 'bifurcations', 'terminals', 
                'branch_density', 'ratio_dend_soma', 'product_soma_dend', 'ratio_bifur_term',
                'ratio_term_dend', 'log_bifurcations', 'log_terminals', 'soma_radius_sq', 
                'dend_length_sq', 'product_bifur_term']

y = data[target_col].values

#reads through the data and creates numpy arrays with dataset dimensions
#reads through the data and creates numpy arrays with dataset dimensions
num_observations = X.shape[0]
num_features = X.shape[1]  # Now 14 instead of 5

print(f"Number of Observations = {num_observations}")
print(f"Number of Features = {num_features}")
print(f"Target (Expression) range: {y.min():.4f} to {y.max():.4f}")
print(f"Target mean ± std: {y.mean():.4f} ± {y.std():.4f}\n")

#encodes brain regions for stratification
region_labels = pd.factorize(data['brain_region'])[0]
unique_regions = len(np.unique(region_labels))
print(f"Number of unique brain regions: {unique_regions}")


# ===== NEURAL NETWORK MODEL =====
#class that creates a custom feedforward network that inherits all of pytorch's training tools
class NeuralNetwork(nn.Module):

    #constructor
    def __init__(self, input_size, hidden_1, hidden_2, output_size=1, dropout=0.2):
        super(NeuralNetwork, self).__init__()
        
        #defines the layers of the network
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_2, output_size)

    #defines what happens when data flows through the model
    def forward(self, x):
        #first hidden layer with ReLU activation and dropout
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        #second hidden layer with ReLU activation and dropout
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        #output layer (linear activation for regression)
        x = self.fc3(x)
        
        return x
# NEW (increase hidden layers since you have more features)
num_features = X.shape[1]  # Now 14 instead of 5
HIDDEN_1 = 128
HIDDEN_2 = 64
print("Model architecture: {} -> {} -> {} -> 1".format(num_features, HIDDEN_1, HIDDEN_2))

print("Model architecture: {} -> {} -> {} -> 1".format(num_features, HIDDEN_1, HIDDEN_2))


# ===== DEVICE CONFIGURATION =====
#determines which device to use for computation (GPU or CPU)
if USE_GPU and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal Performance Shaders (MPS) for acceleration")
elif USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU for acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ===== STRATIFIED K-FOLD SETUP =====
print("\n" + "="*60)
print("STRATIFIED {}-FOLD CROSS-VALIDATION".format(NUM_FOLDS))
print("="*60)

#creates StratifiedKFold object for splitting data
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

#initializes lists to store results across folds
r2_scores = []
rmse_scores = []
mae_scores = []
pearson_r_scores = []
pearson_p_scores = []
best_models = []
training_losses_per_fold = []
validation_losses_per_fold = []


# ===== K-FOLD TRAINING LOOP =====
#iterates through each fold of cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, region_labels)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*60}")
    
    #splits data into training and testing sets for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    #normalizes features using StandardScaler fitted ONLY on training data
    print("Normalizing data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    print("Normalization complete. Data standardized to mean=0, std=1\n")
    
    #converts numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_normalized).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(1).float().to(device)
    X_test_tensor = torch.from_numpy(X_test_normalized).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(1).float().to(device)
    
    #creates dataloaders from tensor datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("DataLoader created successfully!")
    
    #creates the model for this fold
    model = NeuralNetwork(num_features, HIDDEN_1, HIDDEN_2, output_size=1, dropout=DROPOUT)
    model = model.to(device)
    
    #sets the loss function and optimizer
    criterion = nn.SmoothL1Loss() #changed 
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #adds a scheduler ----- added 3/31
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # to minimize validation loss
    factor=0.5,      # reduce LR by half
    patience=10,     # wait 10 epochs before reducing
    )
    
    print(f"Starting training for fold {fold + 1}...")
    
    #begins the training loop
    training_losses = []
    validation_losses = []
    
    for epoch in range(NUM_EPOCHS):
        #training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        #iterates through training batches
        for batch_x, batch_y in train_loader:
            #forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            
            #update model parameters
            optimizer.step()
            
            #accumulates loss for the epoch
            epoch_loss += loss.item()
            num_batches += 1
        
        #calculates average loss for the epoch
        average_epoch_loss = epoch_loss / num_batches
        training_losses.append(average_epoch_loss)
        
        #validation phase: compute loss on test fold during training
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            scheduler.step(val_loss) #added 3/31
        validation_losses.append(val_loss)
        
        #prints progress every 30 epochs
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {average_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f"Training complete. Final train loss: {training_losses[-1]:.6f}, Final val loss: {validation_losses[-1]:.6f}\n")
    
    #evaluation phase: get predictions on test set
    print("Evaluating on test fold...")
    model.eval()
    with torch.no_grad():
        #starts time measurement for inference
        start_time = time.time()
        
        #makes predictions on test data
        y_pred = model(X_test_tensor)
        
        #ends time measurement
        end_time = time.time()
        prediction_time = end_time - start_time
    
    #converts predictions back to numpy
    y_pred_np = y_pred.cpu().numpy().flatten()
    
    #calculates regression metrics
    r2 = r2_score(y_test, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_np))
    mae = mean_absolute_error(y_test, y_pred_np)
    pearson_r, pearson_p = pearsonr(y_test, y_pred_np)
    
    #prints test metrics for this fold
    print(f"Test Metrics (Fold {fold + 1}):")
    print(f"  R²:        {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  Pearson R: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Inference time: {prediction_time:.4f} seconds")
    
    #stores results from this fold
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    pearson_r_scores.append(pearson_r)
    pearson_p_scores.append(pearson_p)
    best_models.append(model)
    training_losses_per_fold.append(training_losses)
    validation_losses_per_fold.append(validation_losses)
    
    #saves model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/fold_{fold}_best.pt")

# ===== CROSS-VALIDATION SUMMARY =====
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)

#calculates mean and standard deviation of metrics across folds
r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
pearson_mean, pearson_std = np.mean(pearson_r_scores), np.std(pearson_r_scores)

#prints cross-validation summary
print(f"\nR² Score:        {r2_mean:.4f} ± {r2_std:.4f}")
print(f"RMSE:            {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"MAE:             {mae_mean:.4f} ± {mae_std:.4f}")
print(f"Pearson R:       {pearson_mean:.4f} ± {pearson_std:.4f}")
print(f"All folds significant (p<0.001): {all(p < 0.001 for p in pearson_p_scores)}")


# ===== SAVE RESULTS =====
print("\nSaving results...")

#creates results dataframe with per-fold metrics
results = pd.DataFrame({
    'fold': range(1, NUM_FOLDS + 1),
    'r2': r2_scores,
    'rmse': rmse_scores,
    'mae': mae_scores,
    'pearson_r': pearson_r_scores,
    'p_value': pearson_p_scores
})

#saves results to CSV
os.makedirs("results", exist_ok=True)
results.to_csv("results/cross_validation_metrics.csv", index=False)
print("✓ Results saved to results/cross_validation_metrics.csv")


# ===== LEARNING CURVES VISUALIZATION =====
print("\nGenerating learning curves...")

#plot: aggregate training/validation loss curves for all folds
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle('Training vs Validation Loss Across All Folds\n(Diverging lines = overfitting; Converging lines = good generalization)', 
             fontsize=16, fontweight='bold', y=0.995)

for fold in range(NUM_FOLDS):
    ax = axes[fold // 5, fold % 5]
    ax.plot(training_losses_per_fold[fold], label='Training', linewidth=2.5, color='steelblue', alpha=0.8)
    ax.plot(validation_losses_per_fold[fold], label='Validation', linewidth=2.5, color='coral', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    ax.set_title(f'Fold {fold+1} (R²={r2_scores[fold]:.3f}, p={pearson_p_scores[fold]:.2e})', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.4, linestyle='--')
    ax.set_xlim([0, NUM_EPOCHS-1])

plt.tight_layout()
os.makedirs("results/learning_curves", exist_ok=True)
plt.savefig("results/learning_curves/all_folds_learning_curves.png", dpi=300, bbox_inches='tight')
print("✓ Saved: results/learning_curves/all_folds_learning_curves.png")
plt.close()

#plot: individual learning curve for each fold (higher resolution)
for fold in range(NUM_FOLDS):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_losses_per_fold[fold], label='Training Loss', linewidth=2.5, color='steelblue', alpha=0.85)
    ax.plot(validation_losses_per_fold[fold], label='Validation Loss', linewidth=2.5, color='coral', alpha=0.85)
    ax.fill_between(range(len(training_losses_per_fold[fold])), training_losses_per_fold[fold], 
                     validation_losses_per_fold[fold], alpha=0.2, color='gray', label='Gap (overfitting indicator)')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'Fold {fold+1} Learning Curve\nR²={r2_scores[fold]:.4f}, RMSE={rmse_scores[fold]:.4f}, Pearson R={pearson_r_scores[fold]:.4f} (p={pearson_p_scores[fold]:.2e})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.4, linestyle='--')
    ax.set_xlim([0, NUM_EPOCHS-1])
    plt.tight_layout()
    plt.savefig(f"results/learning_curves/fold_{fold+1:02d}_learning_curve.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: results/learning_curves/fold_{fold+1:02d}_learning_curve.png")
    plt.close()


# ===== PERFORMANCE VISUALIZATIONS =====
print("\nGenerating performance visualizations...")

#creates list to store all predictions for visualization
all_y_test_list = []
all_y_pred_list = []

#recomputes predictions on all folds for visualization
for fold, (train_idx, test_idx) in enumerate(skf.split(X, region_labels)):
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    #normalizes test data using training scaler
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_test_normalized = scaler.transform(X_test)
    X_test_tensor = torch.from_numpy(X_test_normalized).float().to(device)
    
    #makes predictions using best model from this fold
    model = best_models[fold]
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    
    #stores predictions
    all_y_test_list.extend(y_test)
    all_y_pred_list.extend(y_pred.cpu().numpy().flatten())

#converts to numpy arrays
all_y_test = np.array(all_y_test_list)
all_y_pred = np.array(all_y_pred_list)

#plot 1: R² scores across folds
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(1, NUM_FOLDS + 1), r2_scores, color='steelblue', edgecolor='black', alpha=0.75, linewidth=1.5)
ax.axhline(r2_mean, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {r2_mean:.4f}')
ax.axhline(r2_mean + r2_std, color='red', linestyle=':', linewidth=2, alpha=0.6, label=f'±1 Std: {r2_std:.4f}')
ax.axhline(r2_mean - r2_std, color='red', linestyle=':', linewidth=2, alpha=0.6)
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation R² Scores Across Folds', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, NUM_FOLDS + 1))
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([min(r2_scores)-0.1, max(r2_scores)+0.1])
plt.tight_layout()
plt.savefig("results/cv_r2_scores.png", dpi=300, bbox_inches='tight')
print("✓ Saved: results/cv_r2_scores.png")
plt.close()

#plot 2: Predicted vs Actual
fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(all_y_test, all_y_pred, alpha=0.65, s=80, edgecolors='black', linewidth=0.8, color='steelblue')
min_val = min(all_y_test.min(), all_y_pred.min())
max_val = max(all_y_test.max(), all_y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect prediction (y=x)', alpha=0.8)
ax.set_xlabel('Actual Expression Energy', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Expression Energy', fontsize=12, fontweight='bold')
ax.set_title(f'Predicted vs Actual Expression (All Folds Combined)\nPearson R={pearson_mean:.4f} ± {pearson_std:.4f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("results/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
print("✓ Saved: results/predicted_vs_actual.png")
plt.close()

#plot 3: Residuals
residuals = all_y_test - all_y_pred
fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(all_y_pred, residuals, alpha=0.65, s=80, edgecolors='black', linewidth=0.8, color='steelblue')
ax.axhline(0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
ax.axhline(residuals.std(), color='orange', linestyle=':', linewidth=2, alpha=0.6, label=f'±1 Std: {residuals.std():.4f}')
ax.axhline(-residuals.std(), color='orange', linestyle=':', linewidth=2, alpha=0.6)
ax.set_xlabel('Predicted Expression Energy', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax.set_title(f'Residual Plot\nMAE={mae_mean:.4f} ± {mae_std:.4f}, RMSE={rmse_mean:.4f} ± {rmse_std:.4f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("results/residuals.png", dpi=300, bbox_inches='tight')
print("✓ Saved: results/residuals.png")
plt.close()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nResults saved to: results/")
print(f"Models saved to: models/")
print(f"Learning curves saved to: results/learning_curves/")
