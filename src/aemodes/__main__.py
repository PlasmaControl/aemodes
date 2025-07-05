from pathlib import Path

from .scripts.train import train_model
from .scripts.evaluate import evaluate_model

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import wandb
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(
    version_base=None, 
    config_path="config", 
    config_name="default",
    )
def main(cfg: DictConfig):

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    config_dict = json.loads(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=config_dict,
    )
    run.name = cfg.wandb.id
    print(f"Run Name: {run.name}, Run ID: {run.id}")
    log_fn = lambda metrics: wandb.log(metrics)

    try:
        train_model(cfg=cfg, log_fn=log_fn, output_dir=output_dir)

        # Load best model and generate inference GIF
        evaluate_model(cfg=cfg, output_dir=output_dir)

    finally:
        run.finish()
        
    return

if __name__ == "__main__":
    main()