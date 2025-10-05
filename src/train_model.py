"""
Demo training entrypoint using Hydra config.

This assumes config keys:
  config.data.processed
  config.model.type
  config.model.hyperparameters
  config.model.cross_validation.{folds, scoring}
"""

import hydra
from omegaconf import DictConfig


def run_training(cfg: DictConfig) -> None:
    print("Training model...")
    print(f"Processed data: {cfg.data.processed}")
    print(f"Model type: {cfg.model.type}")
    print(f"Hyperparameters: {cfg.model.hyperparameters}")
    print(f"CV folds: {cfg.model.cross_validation.folds}")
    print(f"Scoring: {cfg.model.cross_validation.scoring}")
    print("Saving model to: models/ (implement persist logic)")
    # TODO: load processed data, fit model, joblib.dump(...)

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
