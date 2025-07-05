import numpy as np
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from ..utils.dataset import ShotDataset, load_dataset
from ..utils import EarlyStopping
from ..nn.loss import BCELoss, DiceLoss, MSELoss, TPRLoss, FPRLoss
import hydra

def train_model(
    cfg,
    log_fn=None,
    output_dir=None,
):
    
    # Load Configuration Objects
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    early_stopper = hydra.utils.instantiate(cfg.early_stopper)
    
    valid_loss = {}
    for name, loss in cfg.validation_loss.items():
        valid_loss[name] = hydra.utils.get_class(loss)()
    epoch_val_losses = {loss_name: 0. for loss_name, losses in valid_loss.items()}
    avg_val_losses = {loss_name: 0. for loss_name, losses in valid_loss.items()}

    # Load dataset
    train_dataset, valid_dataset = load_dataset(cfg.train.data_file)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.train.batch_size, 
        num_workers=cfg.train.num_workers,
        shuffle=False, 
        pin_memory=True,
        persistent_workers=True,
    )

    # Model setup
    model.to(device)

    best_loss = float('inf')

    for epoch in range(cfg.train.epochs):
        # Reset epoch_val_losses at the start of each epoch
        epoch_val_losses = {loss_name: 0. for loss_name in valid_loss.keys()}

        # -------- training --------
        model.train()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            feature = batch['X'].to(device)
            label   = batch['y'].to(device)

            prediction = model(feature)
            loss = criterion(prediction, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if log_fn:
                log_fn({"train_loss": loss.item()})

        # -------- validation --------
        model.eval()

        with torch.no_grad():
            for vbatch in valid_loader:
                vfeat  = vbatch['X'].to(device)
                vlab   = vbatch['y'].to(device)

                for loss_name, loss_fn in valid_loss.items():
                    vpred  = model(vfeat)
                    vloss  = loss_fn(vpred, vlab)
                    epoch_val_losses[loss_name] += vloss.item()

        for loss_name, losses in epoch_val_losses.items():
            avg_val_losses[loss_name] = float(np.mean(losses))

        # Log validation metrics if log_fn is provided
        if log_fn:
            log_fn({f"val_loss_{loss_name}": avg_loss for loss_name, avg_loss in avg_val_losses.items()})

        early_stopper(avg_val_losses["BCE"])  # Use BCE loss for early stopping


        # checkpoint
        if avg_val_losses["BCE"] < best_loss:
            best_loss = avg_val_losses["BCE"]
            torch.save(model.state_dict(), output_dir / cfg.train.ckpt_file)

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break