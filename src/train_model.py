from __future__ import annotations

import json
from pathlib import Path
from posixpath import abspath
from typing import Sequence
import hydra
import joblib
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from typing import Any, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, auc

from src.utils import _resolve_repo_path, load_data, make_estimator

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    run_training(config)

def run_training(config: DictConfig) -> None:
    #  Load processed/final training data 

    X = load_data("./data/final/X_train_final.csv")
    X = X.loc[:, ~X.columns.str.contains("^Unnamed")]
    y = load_data("./data/final/y_train_final.csv").squeeze("columns")

    # Train/valid split (stratified) 
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)


    print("Training model...")
    print(f"Processed data: {config.data.processed}")
    print(f"Model type: {config.model.type}")
    print(f"Hyperparameters: {config.model.hyperparameters}")

    # Build estimator and param grid 
    estimator: ClassifierMixin = make_estimator(str(config.model.type))
    #param_grid = _to_param_grid(dict(config.model.hyperparameters))
    param_grid = dict(config.model.hyperparameters)

    # Run hyperparameter search on the split 
    results = hyper_search(   # uses the function we built previously
        estimator=estimator,
        param_grid=param_grid,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_val,
        y_test=y_val,
        cv=int(config.model.cross_validation.folds),
        plot_curve="prc",  # change to "roc" if you prefer
    )

    # Persist best results 
    best_params = results["best_params"]
    final_model: ClassifierMixin = make_estimator(str(config.model.type))
    # apply the grid-searched params
    final_model.set_params(**best_params)  # type: ignore[attr-defined]
    final_model.fit(X, y)

    # --- Persist best model (fully refit) & results ---
    models_dir = _resolve_repo_path(Path(config.paths.models_dir))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{config.model.type}_best.joblib"
    joblib.dump(final_model, model_path)

    # add the refit detail to results.json
    results_out = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in results.items() if k != "best_model"}
    results_out["refit_on_full_train"] = True
    results_out["best_params"] = best_params
    results_out["model_path"] = str(model_path)

    results_json = _resolve_repo_path(Path("results") / "results.json")
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results_out, indent=2))

    print(f"Saved fully-refit best model to: {model_path}")
    print(f"Saved metrics to:                {results_json}")



def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: Sequence[str] | None = None) -> None:
    
    cm = confusion_matrix(y_true, y_pred)
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    display_labels = labels if labels is not None else [str(c) for c in sorted(np.unique(y_true))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.show()
    print(classification_report(y_true, y_pred, target_names=display_labels, zero_division=0))


def hyper_search(
    estimator: ClassifierMixin,
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    cv: int = 5,
    plot_curve: str = "prc"
    ) -> tuple[ClassifierMixin, np.ndarray, float, float, float, float, float]:
    
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        verbose=0,
        n_jobs=-1,
    )

    gs.fit(X_train, y_train)
    clf: ClassifierMixin = gs.best_estimator_

    y_pred_test: np.ndarray = clf.predict(X_test)  

    if hasattr(clf, "predict_proba"):
        y_proba_test = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_proba_test = clf.decision_function(X_test)
    else:
        y_proba_test = np.zeros_like(y_test, dtype=float)

    # Metrics
    f1_train: float = float(gs.best_score_)
    f1_test: float = float(f1_score(y_test, y_pred_test, average="binary"))
    roc_auc: float = float(roc_auc_score(y_test, y_proba_test))
    precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
    auc_pr: float = float(auc(recall, precision))

    print(f"f1_train: {gs.best_score_:.4f} using {gs.best_params_}")
    print("f1_test:", f1_test)

    plot_confusion_matrix(y_test, y_pred_test)

    # Plot ROC or PRC
    if plot_curve.lower() == "roc":
        plot_roc_curve(y_test, y_proba_test, roc_auc)

    elif plot_curve.lower() == "prc":
        plot_pr_curve(precision, recall, auc_pr)


    metrics_dict = {
        "best_model": clf,
        "best_params": gs.best_params_,
        "f1_train": f1_train,
        "f1_test": f1_test,
        "roc_auc": roc_auc,
        "auc_precision_recall": auc_pr,
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
    }

    return metrics_dict

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, auc_value: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, auc_value: float) -> None:
    plt.plot(recall, precision, label=f"PR curve (AUC = {auc_value:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
