import hydra
from omegaconf import DictConfig

def run_evaluation(cfg: DictConfig) -> None:
    print("📊 Evaluating model...")
    print("Loading from: models/")
    print("Writing metrics to: results/metrics.json (implement evaluation logic)")
    # TODO: load model, load test/holdout, compute metrics, save JSON + plots

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    run_evaluation(cfg)

if __name__ == "__main__":
    main()
