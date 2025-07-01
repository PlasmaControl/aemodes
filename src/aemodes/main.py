import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(
    version_base=None, 
    config_path="config", 
    config_name="default",
    )
def main(cfg: DictConfig):
    print(cfg)
    return

if __name__ == "__main__":
    main()