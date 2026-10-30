import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.datamodules.forest_datamodule import ForestDataModule


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    datamodule = ForestDataModule(**cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
