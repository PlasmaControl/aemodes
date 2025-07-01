

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import DataLoader, Dataset

import pickle
from projects.co2modes.models.seldnet import SELDNet as BaselineModel

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from pathlib import Path


# -----------------------------------------------------------------------------#
# Initialisation
# -----------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShotDataset(Dataset):
    def __init__(self, shots, X, y):
        self.shots = shots
        self.X = X
        self.y = y
        
        lenshot = 3905
        self.nwin = 11
        self.lenwin = lenshot // self.nwin
        self.hoplen = lenshot // self.nwin
    
    def __len__(self):
        return len(self.shots) * self.nwin
    
    def __getitem__(self, idx):
        shot_idx, win_idx = idx // self.nwin, idx % self.nwin
        start_idx = win_idx * self.hoplen
        end_idx = start_idx + self.lenwin
        X = self.X[shot_idx]
        X = torch.tensor(np.stack([X['r0'],X['v1'],X['v2'],X['v3']])[:,start_idx:end_idx])
        y = torch.tensor(self.y[shot_idx][start_idx:end_idx])
        return {
            'shot': self.shots[shot_idx],
            'X': X.float(),
            'y': y.float(),
        }

aza_data = '/scratch/gpfs/nc1514/specseg/data/samples_aza/co2_250_detector.pkl'
    
[train_shots, X_train,y_train,valid_shots, X_valid,y_valid] = pickle.load(open(aza_data,'rb'))

train_dataset = ShotDataset(train_shots, X_train, y_train)
valid_dataset = ShotDataset(valid_shots, X_valid, y_valid)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=2
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=2
)

print(f"Train loader batches: {len(train_loader)}")
print(f"Valid loader batches: {len(valid_loader)}")

# -----------------------------------------------------------------------------#
# Model & Metrics
# -----------------------------------------------------------------------------#
model = BaselineModel().to(device)

checkpoint_path = Path("output/co2cross/best_model.pth")
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint_path}. "
        "Train the model or update the path before running evaluation."
    )
model.eval()

criterion_bce = nn.BCEWithLogitsLoss(reduction="sum")
criterion_mse = nn.MSELoss(reduction="sum")


def evaluate(loader):
    """Return average BCE and MSE loss over a dataloader."""
    total_bce, total_mse, n_samples = 0.0, 0.0, 0
    batchnum = 0
    lenbatch = len(loader)
    with torch.no_grad():
        for batch in loader:
            features = batch["X"].to(device)
            labels = batch["y"].to(device)

            logits = model(features)
            bce = criterion_bce(logits, labels)

            # For MSE, compare probabilities (after sigmoid) with labels
            probs = torch.sigmoid(logits)
            mse = criterion_mse(probs, labels)

            total_bce += bce.item()
            total_mse += mse.item()
            n_samples += labels.size(0)
            
            # Print progress every 10 batches
            batchnum += 1
            if batchnum % 10 == 0:
                print(f"Evaluating batch {batchnum} of {lenbatch}...", flush=True)
                print(f"  - BCE Loss: {total_bce / n_samples:.6f}", flush=True)
                print(f"  - MSE Loss: {total_mse / n_samples:.6f}", flush=True)

    avg_bce = total_bce / n_samples
    avg_mse = total_mse / n_samples
    return avg_bce, avg_mse


# -----------------------------------------------------------------------------#
# Run evaluation & report
# -----------------------------------------------------------------------------#

print("\n---------------------  Evaluation  ---------------------")
train_bce, train_mse = evaluate(train_loader)
print(f"Training   | BCE Loss: {train_bce:.6f} | MSE Loss: {train_mse:.6f}", flush=True)
valid_bce, valid_mse = evaluate(valid_loader)
print(f"Validation | BCE Loss: {valid_bce:.6f} | MSE Loss: {valid_mse:.6f}", flush=True)
print("--------------------------------------------------------\n")