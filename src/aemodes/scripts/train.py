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

from ..utils.dataset import ShotDataset
from models.baseline import Model


datapath = Path('/scratch/gpfs/nc1514/specseg/data/samples_aza/co2_250_detector.pkl')
[train_shots, X_train,y_train,valid_shots, X_valid,y_valid] = pickle.load(open(datapath,'rb'))

train_dataset = ShotDataset(train_shots, X_train, y_train)
valid_dataset = ShotDataset(valid_shots, X_valid, y_valid)

batch_size = 32
num_workers = 3

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers,
    shuffle=False, 
    pin_memory=True,
    persistent_workers=True,
)

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

model = Model().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
early_stopper = EarlyStopping(patience=2, delta=1e-4)

epochs = 100
losses = []
valid_losses = []
train_steps = []
valid_steps = []
best_loss = float('inf')

output_dir = Path('output/co2cross')
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