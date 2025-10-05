from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

from src.train_model import plot_confusion_matrix, plot_pr_curve, plot_roc_curve

from .utils import load_data, _resolve_repo_path

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    run_predict(config)


def run_predict(config: DictConfig) -> None:
    final_dir = Path(config.data.final)
    X_path = _resolve_repo_path(final_dir / "X_test_final.csv")
    y_path = _resolve_repo_path(final_dir / "y_test_final.csv")

    models_dir = _resolve_repo_path(Path(config.paths.models_dir))
    model_path = models_dir / f"{config.model.type}_best.joblib"

    # Load data 
    X_te = load_data(X_path)
    X_te = X_te.loc[:, ~X_te.columns.str.contains("^Unnamed")]
    y_te = load_data(y_path).squeeze("columns").astype("int8").to_numpy()

    # Load model 
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}. Train first.")
    model: ClassifierMixin = joblib.load(model_path)

    # Predict 
    y_pred = model.predict(X_te)  # type: ignore[assignment]
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_te)
    else:
        print("Model has no probability or decision_function method; using zeros for scores.\n")
        y_score = np.zeros_like(y_te, dtype=float)

    # Metrics
    precision, recall, _ = precision_recall_curve(y_te, y_score)
    metrics_dict: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_te, y_score)),
        "auc_precision_recall": float(np.trapz(precision, recall)),  # auc(recall, precision)
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "model_path": str(model_path),
        "n_samples": int(len(y_te)),
    }

    print("Test metrics:", json.dumps(metrics_dict, indent=2))

    # Plots
    plot_confusion_matrix(y_te, y_pred)
    if str(config.training.plot_curve).lower() == "roc":
        plot_roc_curve(y_te, y_score, metrics_dict["roc_auc"])
    else:
        plot_pr_curve(precision, recall, metrics_dict["auc_precision_recall"])

    # Persist predictions + metrics 
    results_dir = _resolve_repo_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    preds_path = results_dir / "predictions.csv"
    metrics_path = results_dir / "prediction_metrics.json"

    preds_df = pd.DataFrame(
        {"y_true": y_te, "y_pred": y_pred, "y_score": y_score},
        index=X_te.index if X_te.index is not None else None,
    )
    preds_df.to_csv(preds_path, index=True)

    metrics_path.write_text(json.dumps(metrics_dict, indent=2))

    print(f"Saved predictions to: {preds_path}")
    print(f"Saved metrics to:     {metrics_path}")


if __name__ == "__main__":
    main()
