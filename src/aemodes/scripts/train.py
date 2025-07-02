import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.style.use('dark_background')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from ..utils.dataset import ShotDataset, load_dataset
from ..utils import EarlyStopping

def train_model(
    model,
    optimizer,
    criterion,
    early_stopper,
    cfg,
    ):
    
    # Load dataset
    train_dataset, valid_dataset = load_dataset(cfg.data_file)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        shuffle=False, 
        pin_memory=True,
        persistent_workers=True,
    )

    # Model setup
    model.to(device)

    epochs = cfg.epochs
    losses = []
    valid_losses = []
    train_steps = []
    valid_steps = []
    best_loss = float('inf')

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader_len = len(train_loader)

    for epoch in range(epochs):
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

            if idx % 5 == 0:
                losses.append(loss.item())
                train_steps.append(epoch * loader_len + idx)

        # -------- validation --------
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for vbatch in valid_loader:
                vfeat  = vbatch['X'].to(device)
                vlab   = vbatch['y'].to(device)
                vpred  = model(vfeat)
                vloss  = criterion(vpred, vlab)
                val_epoch_losses.append(vloss.item())

        avg_val_loss = float(np.mean(val_epoch_losses))
        valid_losses.append(avg_val_loss)
        valid_steps.append((epoch + 1) * loader_len)

        early_stopper(avg_val_loss)

        # plot
        plt.figure()
        plt.plot(train_steps, losses, marker='o', label='train')
        plt.plot(valid_steps, valid_losses, marker='x', label='valid')
        plt.ylim(0, max(max(losses), max(valid_losses)) * 1.1)
        plt.title(f'Epoch {epoch + 1} | val {avg_val_loss:.4f}')
        plt.legend()
        plt.savefig(output_dir / 'training_loss.png')
        plt.close()

        # checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break