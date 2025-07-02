from .scripts.train import train_model

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(
    version_base=None, 
    config_path="config", 
    config_name="default",
    )
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    early_stopper = hydra.utils.instantiate(cfg.early_stopper)
    train_model(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        early_stopper=early_stopper,
        cfg=cfg.train
        )
    # evaluate_model(cfg)
    # inference_model(cfg)
    return

if __name__ == "__main__":
    main()