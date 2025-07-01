import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torchvision.transforms as TV
from tqdm.auto import tqdm
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader

from specseg.core.dataset import SignalAnalysisDataset, SignalMLDataset
from specseg.core.transform import Transform
from specseg.core import visualize as vis
from specseg.transform.co2cross import HFLabel
from specseg.transform.co2cross import HFFeature as CrossFeature
from specseg import helpers

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from models.backbone import dino
from models.co2 import CO2AE1D03 as CO2Model

import pandas as pd


sns.set_style("dark")
plt.style.use('dark_background')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

backbone = dino(2)
model = CO2Model(backbone, 5, 150)
model = model.to(device)

config_name = "co2_small"

if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
initialize(version_base=None, config_path="config/dataset", job_name="early_fusion_co2")
cfg = compose(config_name=config_name)
cfg = OmegaConf.to_container(cfg, resolve=True)

root_dir = cfg['dataset_cfg']['root_dir']
train_shots = helpers.get_shots(root_dir, 'train')
val_shots = helpers.get_shots(root_dir, 'valid')

cross_inputs = ['r0', 'v1', 'v2', 'v3']
labels = ['lfm','bae','eae', 'rsae', 'tae']

transform = Transform(
    CrossFeature,
    HFLabel,
    cfg=cfg['transform_cfg'],
)

dataset_train = SignalMLDataset(
    stage='train',
    inputs=cross_inputs,
    labels=labels,
    shots=train_shots,
    transform=transform,
    cfg=cfg['dataset_cfg'],
)
dataset_train.setup()

dataset_val = SignalMLDataset(
    stage='valid',
    inputs=cross_inputs,
    labels=labels,
    shots=val_shots,
    transform=transform,
    cfg=cfg['dataset_cfg'],
)
dataset_val.setup()

datpath = Path(dataset_train.cfg['root_dir']) / 'train' / '170805.parquet'

data = pd.read_parquet(datpath).time.values
tstart = data[0] / np.timedelta64(1, 'ms')
tend = data[-1] / np.timedelta64(1, 'ms')

dataset_train.location(110*18+4)

dataset_train.transform.transform_label.cfg['nframes'] = model.resolution

import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt  = torch.exp(-bce)
    loss = alpha * (1-pt)**gamma * bce
    return loss.mean()

for p in model.backbone.parameters():
    p.requires_grad = False
    
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.head.parameters())

test_input = torch.randn(1,3,518,518).to(device)
with torch.no_grad():
    output = model(test_input)
print(output.shape)

batch_size = 32
num_workers = 1
prefetch_factor = 2

loader = DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    shuffle=True, 
    pin_memory=False,
    persistent_workers=True,
    )

print(len(loader))

epochs = 1
losses = []

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')

for epoch in range(epochs):
    for idx, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        feature = batch['features'].to(device)
        label = batch['labels'].to(device)
        prediction = model(feature)
        loss = criterion(prediction, label)
        loss.backward();  optimizer.step()
        
torch.save(model.state_dict(), 'mdl1.pt')