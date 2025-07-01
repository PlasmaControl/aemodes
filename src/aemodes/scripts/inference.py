import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns
from torch.utils.data import DataLoader

from specseg.core.dataset import SignalAnalysisDataset, SignalMLDataset
from specseg.core.transform import Transform
from specseg.transform.co2cross import HFFeature as CrossFeature
from specseg import helpers
from specseg.core import visualize as vis

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from models.backbone import dino
from models.co2 import CO2AE1D03 as CO2Model

sns.set_style("dark")
plt.style.use('dark_background')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Model
backbone = dino(2)
model = CO2Model(backbone, 5, 150)
model = model.to(device)

state_dict = torch.load(
    "/scratch/gpfs/nc1514/specseg/mdl.pt", 
    weights_only=True, 
    map_location=device
    )

model.load_state_dict(state_dict)
model.eval()

print("Model Ready")


# Config
config_name = "co2_small_infer"

if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
initialize(version_base=None, config_path="config/dataset", job_name="early_fusion_co2_infer")
cfg = compose(config_name=config_name)
cfg = OmegaConf.to_container(cfg, resolve=True)

stage = 'train'
root_dir = cfg['dataset_cfg']['root_dir']
shots = helpers.get_shots(root_dir, stage)

cross_inputs = ['r0', 'v1', 'v2', 'v3']
labels = ['lfm','bae','eae', 'rsae', 'tae']

transform = Transform(
    CrossFeature,
    cfg=cfg['transform_cfg'],
)

dataset = SignalAnalysisDataset(
    stage=stage,
    inputs=cross_inputs,
    shots=shots,
    transform=transform,
    cfg=cfg['dataset_cfg'],
)
dataset.setup()


# Infer
dataset.on_off = False
loader = DataLoader(dataset, batch_size=10)

B, H, W = 10, 5, 150
result_dir_root = Path('/scratch/gpfs/nc1514/specseg/output/co2cross/inference')
windows_per_shot = dataset.cfg['window_count']
for idx, data in enumerate(loader):
    x = data['features']
    shot_num = dataset.shots[idx]
    
    # idx = 1201
    # x = []
    # for i in range(10):
    #     xtemp = dataset[idx*10+i]['features']
    #     x.append(xtemp)
    # x = torch.stack(x, dim=0)
    # x = x.to(device)
    
    with torch.no_grad():
        result = model(x.to(device))
        result = torch.sigmoid(result)
    result2 = result.permute(1,0,2).reshape(H, B * W)
    result2 = result2.cpu().numpy()
    result_dir = result_dir_root / str(shot_num)
    np.save(result_dir, result2)
    
    Bimg, Cimg, Himg, Wimg = x.shape
    stacked = x.permute(1, 2, 0, 3).reshape(Cimg, Himg, Bimg * Wimg)
    img = stacked.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(img, aspect='auto', cmap='magma', origin='lower')
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(result2 > 0.4, aspect='auto', origin='lower', interpolation='none')
    plt.axis('off')
    im_dir = result_dir.with_suffix('.png')
    plt.savefig(im_dir)
    plt.close()
    print(f"Saved {result_dir}", flush=True)