#!/usr/bin/env python
"""
Improve Dataset Labels using SELDNet Models

This script trains 5 separate SELDNet models (one per label class) on noisy labels,
then uses the trained models to generate improved/smoothed labels for the dataset.

Intermediate results are cached to data/.cache/improve_labels for resumability.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from aemodes.models.detection.seldnet import SELDNetModel

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_DATA_PATH = Path('/scratch/gpfs/nc1514/aemodes/data/co2_250_detector.pkl')
DEFAULT_OUTPUT_PATH = Path('/scratch/gpfs/nc1514/aemodes/data/co2_250_detector_2.pkl')
DEFAULT_CACHE_DIR = Path('/scratch/gpfs/nc1514/aemodes/data/.cache/improve_labels')

# Training settings
NUM_LABELS = 5
NUM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
THRESHOLD = 0.5  # For inference
SMOOTH_KERNEL = 5  # Temporal smoothing kernel size
GPUS = 1

# =============================================================================
# Loss Function
# =============================================================================

class BinarySCELoss(nn.Module):
    """
    Binary Symmetric Cross Entropy Loss for noisy label training.
    
    Adapted from https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    for binary classification with sigmoid outputs.
    
    SCE = alpha * BCE + beta * RCE
    - BCE: standard binary cross entropy
    - RCE: reverse cross entropy (labels predicting model outputs)
    """
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred_logits, targets):
        """
        Args:
            pred_logits: Raw logits from model (B, T) or (B, T, 1)
            targets: Binary labels (B, T) with values in {0, 1}
        """
        # Flatten if needed
        pred_logits = pred_logits.view(-1)
        targets = targets.view(-1).float()
        
        # Get probabilities
        pred_prob = torch.sigmoid(pred_logits)
        
        # Clamp for numerical stability
        pred_prob = torch.clamp(pred_prob, min=1e-7, max=1.0 - 1e-7)
        targets_clamped = torch.clamp(targets, min=1e-4, max=1.0 - 1e-4)
        
        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='mean')
        
        # RCE: -[p*log(y) + (1-p)*log(1-y)]
        rce = -(pred_prob * torch.log(targets_clamped) + 
                (1 - pred_prob) * torch.log(1 - targets_clamped))
        rce = rce.mean()
        
        # Combined loss
        loss = self.alpha * bce + self.beta * rce
        return loss


# =============================================================================
# Dataset and DataModule
# =============================================================================

class SingleLabelShotDataset(Dataset):
    """
    Dataset that extracts a single label class for per-class training.
    Based on ShotDataset from aemodes.utils.dataset but for single-label binary classification.
    """
    def __init__(self, shots, X, y, label_idx):
        """
        Args:
            shots: List of shot identifiers
            X: List of dicts with keys 'r0', 'v1', 'v2', 'v3'
            y: List of arrays with shape (T, num_labels)
            label_idx: Which label index to extract (0-4)
        """
        self.shots = shots
        self.X = X
        self.y = y
        self.label_idx = label_idx
        
        # Window parameters (matching original ShotDataset)
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
        
        # Stack 4 channels: (4, T, F)
        X_dict = self.X[shot_idx]
        X = np.stack([
            X_dict['r0'][start_idx:end_idx],
            X_dict['v1'][start_idx:end_idx],
            X_dict['v2'][start_idx:end_idx],
            X_dict['v3'][start_idx:end_idx]
        ])
        X = torch.tensor(X, dtype=torch.float32)
        
        # Get single label: (T,)
        y_full = self.y[shot_idx][start_idx:end_idx]  # (T, num_labels)
        y = torch.tensor(y_full[:, self.label_idx], dtype=torch.float32)
        
        return {
            'shot': self.shots[shot_idx],
            'X': X,
            'y': y,
        }


class SingleLabelDataModule(L.LightningDataModule):
    """Lightning DataModule for single-label training."""
    
    def __init__(
        self,
        train_shots, X_train, y_train,
        valid_shots, X_valid, y_valid,
        label_idx,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_shots = train_shots
        self.X_train = X_train
        self.y_train = y_train
        self.valid_shots = valid_shots
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = SingleLabelShotDataset(
            self.train_shots, self.X_train, self.y_train, self.label_idx
        )
        self.valid_dataset = SingleLabelShotDataset(
            self.valid_shots, self.X_valid, self.y_valid, self.label_idx
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


# =============================================================================
# SELDNet Lightning Module
# =============================================================================

def lightweight_params():
    """Lightweight SELDNet parameters (~500K params instead of ~5M)."""
    return {
        'pool_sizes': [9, 8, 2],
        'conv_channels': 24,      # Reduced from 64
        'dropout_rate': 0.1,      # Add some regularization
        'nb_cnn2d_filt': 24,      # Reduced from 64
        'rnn_sizes': [64, 64],    # Reduced from [128, 128]
        'fnn_sizes': [64],        # Reduced from [128]
    }


class SELDNetLightningModule(L.LightningModule):
    """Lightning module for SELDNet with Binary SCE Loss."""
    
    def __init__(
        self,
        input_size=(4, 355, 128),
        learning_rate=1e-3,
        sce_alpha=1.0,
        sce_beta=0.5,
        num_epochs=NUM_EPOCHS,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Use lightweight parameters (~500K params)
        params = lightweight_params()
        
        # Build model for single-label output
        self.model = SELDNetModel(
            input_size=input_size,
            output_size=(input_size[1], 1),  # (T, 1) - single label
            params=params,
        )
        
        # Binary SCE Loss
        self.criterion = BinarySCELoss(alpha=sce_alpha, beta=sce_beta)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X = batch['X']  # (B, 4, T, F)
        y = batch['y']  # (B, T)
        
        logits = self.model(X)  # (B, T, 1)
        logits = logits.squeeze(-1)  # (B, T)
        
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X = batch['X']
        y = batch['y']
        
        logits = self.model(X).squeeze(-1)
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# =============================================================================
# Training Function
# =============================================================================

def train_label_model(
    label_idx,
    train_shots, X_train, y_train,
    valid_shots, X_valid, y_valid,
    cache_dir,
    force=False,
):
    """
    Train a SELDNet model for a single label class.
    
    Returns the trained model and checkpoint path.
    """
    model_path = cache_dir / f'model_label_{label_idx}.ckpt'
    
    # Check if cached model exists
    if model_path.exists() and not force:
        print(f"\n{'='*60}")
        print(f"Loading cached model for Label {label_idx}")
        print(f"{'='*60}")
        model = SELDNetLightningModule.load_from_checkpoint(model_path)
        model.eval()
        return model, model_path
    
    print(f"\n{'='*60}")
    print(f"Training model for Label {label_idx}")
    print(f"{'='*60}")
    
    # Create data module
    data_module = SingleLabelDataModule(
        train_shots, X_train, y_train,
        valid_shots, X_valid, y_valid,
        label_idx=label_idx,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    
    # Determine input size from data
    data_module.setup()
    sample = data_module.train_dataset[0]
    input_size = tuple(sample['X'].shape)  # (4, T, F)
    print(f"Input size: {input_size}")
    
    # Create model
    model = SELDNetLightningModule(
        input_size=input_size,
        learning_rate=LEARNING_RATE,
        sce_alpha=1.0,
        sce_beta=0.5,
        num_epochs=NUM_EPOCHS,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cache_dir,
        filename=f'model_label_{label_idx}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',
        devices=GPUS,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        precision='bf16-mixed',
        logger=False,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Load best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved to: {best_model_path}")
    
    # Load and return the best model
    best_model = SELDNetLightningModule.load_from_checkpoint(best_model_path)
    best_model.eval()
    
    return best_model, best_model_path


# =============================================================================
# Inference Functions
# =============================================================================

def temporal_smooth(predictions, kernel_size=5):
    """
    Apply temporal smoothing using a moving average filter.
    Helps clean up noisy per-frame predictions.
    """
    if kernel_size <= 1:
        return predictions
    
    # Pad and convolve
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(predictions, kernel, mode='same')
    return smoothed


def run_inference_on_shot(model, X_dict, device, lenshot=3905, nwin=11, threshold=0.5, smooth_kernel=5):
    """
    Run inference on a single shot using a single trained model.
    
    Returns:
        predictions: Array of shape (T,) with binary predictions for one label
    """
    lenwin = lenshot // nwin
    hoplen = lenshot // nwin
    
    # Accumulate predictions across windows
    all_predictions = np.zeros(lenshot)
    counts = np.zeros(lenshot)
    
    model = model.to(device)
    
    for win_idx in range(nwin):
        start_idx = win_idx * hoplen
        end_idx = start_idx + lenwin
        
        # Prepare input: (1, 4, T, F)
        X = np.stack([
            X_dict['r0'][start_idx:end_idx],
            X_dict['v1'][start_idx:end_idx],
            X_dict['v2'][start_idx:end_idx],
            X_dict['v3'][start_idx:end_idx]
        ])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        X = X.to(device)
        
        with torch.no_grad():
            logits = model(X).squeeze()  # (T,)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        all_predictions[start_idx:end_idx] += probs
        counts[start_idx:end_idx] += 1
    
    # Average overlapping predictions
    counts = np.maximum(counts, 1)
    all_predictions = all_predictions / counts
    
    # Apply temporal smoothing
    all_predictions = temporal_smooth(all_predictions, kernel_size=smooth_kernel)
    
    # Threshold to binary
    binary_predictions = (all_predictions > threshold).astype(np.float32)
    
    return binary_predictions


def run_inference_for_label(
    model,
    X_data,
    shots,
    device,
    threshold=THRESHOLD,
    smooth_kernel=SMOOTH_KERNEL,
):
    """
    Run inference for a single label across all shots.
    
    Returns:
        List of 1D arrays, one per shot
    """
    predictions = []
    
    for idx in tqdm(range(len(shots)), desc="Inference"):
        X_dict = X_data[idx]
        lenshot = X_dict['r0'].shape[0]
        
        pred = run_inference_on_shot(
            model,
            X_dict,
            device,
            lenshot=lenshot,
            threshold=threshold,
            smooth_kernel=smooth_kernel,
        )
        predictions.append(pred)
    
    return predictions


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improve dataset labels using SELDNet models')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA_PATH,
                        help='Path to input pickle file')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH,
                        help='Path to output pickle file')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                        help='Directory for caching intermediate results')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if cached model exists')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # Load the dataset
    print(f"\nLoading dataset from: {args.data}")
    with open(args.data, 'rb') as f:
        train_shots, X_train, y_train, valid_shots, X_valid, y_valid = pickle.load(f)
    
    print(f"Train shots: {len(train_shots)}")
    print(f"Valid shots: {len(valid_shots)}")
    print(f"X_train[0] keys: {X_train[0].keys()}")
    print(f"X_train[0]['r0'] shape: {X_train[0]['r0'].shape}")
    print(f"y_train[0] shape: {y_train[0].shape}")
    
    # ==========================================================================
    # Loop over each label: train model, run inference, save predictions
    # ==========================================================================
    
    for label_idx in range(NUM_LABELS):
        preds_path = args.cache_dir / f'preds_label_{label_idx}.pkl'
        
        # Check if predictions already cached
        if preds_path.exists() and not args.force:
            print(f"\n{'='*60}")
            print(f"Predictions for Label {label_idx} already cached, skipping...")
            print(f"{'='*60}")
            continue
        
        # Train model (or load from cache)
        model, model_path = train_label_model(
            label_idx,
            train_shots, X_train, y_train,
            valid_shots, X_valid, y_valid,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        
        # Run inference on training data
        print(f"\nRunning inference for Label {label_idx} on training data...")
        y_train_pred = run_inference_for_label(
            model, X_train, train_shots, device,
            threshold=THRESHOLD, smooth_kernel=SMOOTH_KERNEL,
        )
        
        # Run inference on validation data
        print(f"Running inference for Label {label_idx} on validation data...")
        y_valid_pred = run_inference_for_label(
            model, X_valid, valid_shots, device,
            threshold=THRESHOLD, smooth_kernel=SMOOTH_KERNEL,
        )
        
        # Save predictions to cache
        print(f"Saving predictions for Label {label_idx} to: {preds_path}")
        with open(preds_path, 'wb') as f:
            pickle.dump((y_train_pred, y_valid_pred), f)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # Combine all cached predictions into final output
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Combining predictions from all labels...")
    print(f"{'='*60}")
    
    # Load all cached predictions
    all_train_preds = []
    all_valid_preds = []
    
    for label_idx in range(NUM_LABELS):
        preds_path = args.cache_dir / f'preds_label_{label_idx}.pkl'
        print(f"Loading predictions for Label {label_idx} from: {preds_path}")
        
        with open(preds_path, 'rb') as f:
            y_train_pred, y_valid_pred = pickle.load(f)
        
        all_train_preds.append(y_train_pred)
        all_valid_preds.append(y_valid_pred)
    
    # Combine predictions: stack along label axis
    # all_train_preds[label_idx][shot_idx] -> (T,)
    # We want y_train_improved[shot_idx] -> (T, NUM_LABELS)
    
    y_train_improved = []
    for shot_idx in range(len(train_shots)):
        shot_preds = np.stack([
            all_train_preds[label_idx][shot_idx]
            for label_idx in range(NUM_LABELS)
        ], axis=-1)  # (T, NUM_LABELS)
        y_train_improved.append(shot_preds)
    
    y_valid_improved = []
    for shot_idx in range(len(valid_shots)):
        shot_preds = np.stack([
            all_valid_preds[label_idx][shot_idx]
            for label_idx in range(NUM_LABELS)
        ], axis=-1)  # (T, NUM_LABELS)
        y_valid_improved.append(shot_preds)
    
    # Verify shapes
    print("\nVerifying data shapes...")
    print(f"Original y_train[0] shape: {y_train[0].shape}")
    print(f"Improved y_train[0] shape: {y_train_improved[0].shape}")
    print(f"Original y_valid[0] shape: {y_valid[0].shape}")
    print(f"Improved y_valid[0] shape: {y_valid_improved[0].shape}")
    
    # Compare statistics
    print("\nLabel statistics comparison:")
    for label_idx in range(NUM_LABELS):
        orig_train_mean = np.mean([y[:, label_idx].mean() for y in y_train])
        impr_train_mean = np.mean([y[:, label_idx].mean() for y in y_train_improved])
        orig_valid_mean = np.mean([y[:, label_idx].mean() for y in y_valid])
        impr_valid_mean = np.mean([y[:, label_idx].mean() for y in y_valid_improved])
        
        print(f"  Label {label_idx}:")
        print(f"    Train - Original: {orig_train_mean:.4f}, Improved: {impr_train_mean:.4f}")
        print(f"    Valid - Original: {orig_valid_mean:.4f}, Improved: {impr_valid_mean:.4f}")
    
    # Save improved dataset
    print(f"\nSaving improved dataset to: {args.output}")
    improved_data = [
        train_shots,
        X_train,
        y_train_improved,
        valid_shots,
        X_valid,
        y_valid_improved,
    ]
    
    with open(args.output, 'wb') as f:
        pickle.dump(improved_data, f)
    
    print(f"File size: {args.output.stat().st_size / 1e6:.2f} MB")
    
    # Verify the saved file
    print("\nVerifying saved file...")
    with open(args.output, 'rb') as f:
        loaded_data = pickle.load(f)
    
    loaded_train_shots, loaded_X_train, loaded_y_train, \
    loaded_valid_shots, loaded_X_valid, loaded_y_valid = loaded_data
    
    print(f"Loaded train shots: {len(loaded_train_shots)}")
    print(f"Loaded valid shots: {len(loaded_valid_shots)}")
    print(f"Loaded y_train[0] shape: {loaded_y_train[0].shape}")
    print(f"Loaded y_valid[0] shape: {loaded_y_valid[0].shape}")
    
    print(f"\n{'='*60}")
    print("Dataset improvement complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

