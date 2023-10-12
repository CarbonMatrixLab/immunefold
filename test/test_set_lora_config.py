import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from carbonmatrix.trainer.model_align import set_lora_config

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg : DictConfig):
    set_lora_config(cfg.model, lora_r_seq=16, lora_r_pair=8)

    print(cfg.model)

if __name__ == '__main__':
    main()
