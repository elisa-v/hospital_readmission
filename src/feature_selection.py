# src/feature_selection.py
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.utils.validation import check_is_fitted

from src.utils import _resolve_repo_path, make_estimator, save_dataframe, save_feature_names_as_txt

class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator,
        n_features: int,
        method: str = "SFS",           # "SFS" (forward) or "SBS" (backward)
        corr_threshold: Optional[float] = 0.85,  # None to disable
        scoring: str = "accuracy",
        cv: int = 3,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.method = method
        self.corr_threshold = corr_threshold
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def _correlation_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.corr_threshold is None:
            self._dropped_corr_ = []
            print("Correlation-based filtering excluded in config\n")
            return X
        
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cont_cols = [c for c in num_cols if X[c].nunique(dropna=True) > 2]  # exclude dummies/binaries
        zvar = [c for c in cont_cols if X[c].std(skipna=True) == 0]
        cont_cols = [c for c in cont_cols if c not in zvar]

        if zvar:
            print(f"Zero-variance features removed before corr: {zvar}\n")
        print(f"Numeric columns considered for corr: {len(cont_cols)} \n")

        correlation_matrix = X[cont_cols].corr(numeric_only=True).abs()
        correlation_matrix_no_diago = correlation_matrix.copy()
        np.fill_diagonal(correlation_matrix_no_diago.values, 0.0)
        mean_corr = correlation_matrix_no_diago.mean(axis=0) #mean_corr = corr.mean(axis=0)

        correlated_features = set()
        cols = correlation_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i):
                if correlation_matrix_no_diago.iloc[i, j] > self.corr_threshold:
                    keep_j = mean_corr.iloc[j]
                    keep_i = mean_corr.iloc[i]
                    # drop the one more correlated on average (keep the “less entangled” one)
                    correlated_features.add(cols[i] if keep_i > keep_j else cols[j])
        self._dropped_corr_ = sorted(correlated_features)

        print(f"Highly correlated features: {len(self._dropped_corr_)}. Features to drop:")
        print(", ".join(self._dropped_corr_), "\n")

        self._kept_after_corr_ = [c for c in X.columns if c not in self._dropped_corr_]
        print(f"Features kept after correlation filter ({len(self._kept_after_corr_)}):")
        print(", ".join(self._kept_after_corr_), "\n")

        return X.drop(columns=self._dropped_corr_, errors="ignore")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Work on a copy and remember input column names
        self._input_features_ = X.columns.to_list()
        Xf = self._correlation_filter(X)
        n_avail = Xf.shape[1]
        if n_avail < 2:
            raise ValueError(f"Not enough features after correlation filter: {n_avail}")
        
        if self.method is None or str(self.method).lower() == "none":
            print("Skipping Sequential Feature Selection (method=None).\n")
            kept = Xf.columns.tolist()

        else:
            actual_k = min(self.n_features, n_avail - 1)
            if actual_k != self.n_features:
                warnings.warn(
                    f"Requested n_features={self.n_features} but only {n_avail} available "
                    f"after filtering; using k={actual_k}.\n"
                )

            direction = "forward" if self.method.upper() == "SFS" else "backward"
            sfs = SequentialFeatureSelector(
                self.estimator,
                n_features_to_select=actual_k,
                direction=direction,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs
            )

            print(f"Running Sequential Feature Selection ({direction}, {actual_k} features)...")
            sfs.fit(Xf, y)
            mask = sfs.get_support()
            kept = list(Xf.columns[mask])

        # Save final ordered feature list relative to original input
        self._selected_features_ = kept
        print(f"Final features: \n {self._selected_features_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_selected_features_")
        # Drop corr-removed columns if present; then select the SFS/SBS set
        X2 = X.drop(columns=getattr(self, "_dropped_corr_", []), errors="ignore")
        return X2[self._selected_features_]

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, "_selected_features_")
        return np.array(self._selected_features_)


def run_feature_selection(
    config,
    X_train_prepared: pd.DataFrame,
    y_train: pd.Series) -> Dict[str, Any]:

    fs_cfg = config.feature_selection
    processed_dir = Path(config.data.processed)
    models_dir = Path("models")
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(processed_dir / "X_train_all.csv", X_train_prepared)
    save_dataframe(processed_dir / "y_train.csv", y_train)
    save_feature_names_as_txt(processed_dir, X_train_prepared)

    artifacts: Dict[str, Any] = {
        "all_features": {
            "X_train": str(processed_dir / "X_train_all_features.csv"),
            "y_train": str(processed_dir / "y_train.csv"),
            "n_features": int(X_train_prepared.shape[1]),
        },
        "variants": {},
    }

    if not getattr(fs_cfg, "run", False):
        return artifacts

    est = make_estimator(fs_cfg.estimator)

    for method in fs_cfg.methods:
        if method == "none":
            continue

        for k in fs_cfg.k_list:
            selector = FeatureSelectorTransformer(
                estimator=est,
                n_features=int(k),
                method=str(method),
                corr_threshold=fs_cfg.corr_threshold,
                scoring=fs_cfg.scoring,
                cv=int(fs_cfg.cv),
                n_jobs=int(fs_cfg.n_jobs),
            )

            print(f"\n→ Running Feature Selection: {method} with {k} features using {est}")
            Xtr_k = selector.fit_transform(X_train_prepared, y_train)
            tag = f"{method}_{k}_{est}"

            artifacts["variants"][tag] = save_fs_outputs(
                tag, processed_dir, models_dir, selector, 
                pd.DataFrame(Xtr_k, index=X_train_prepared.index, columns=selector.get_feature_names_out())
            )

    return artifacts

def save_fs_outputs(
    tag: str,
    processed_dir: Path,
    models_dir: Path,
    selector: FeatureSelectorTransformer,
    Xtr: pd.DataFrame) -> Dict[str, Any]:

    paths = {
        "X_train": _resolve_repo_path(processed_dir / f"X_train_{tag}.csv"),
        "feature_names": _resolve_repo_path(processed_dir / f"feature_names_{tag}.txt"),
        "selector": _resolve_repo_path(models_dir / f"selector_{tag}.joblib"),
    }

    save_dataframe(paths["X_train"], Xtr)
    paths["feature_names"].write_text("\n".join(selector.get_feature_names_out()))
    joblib.dump(selector, paths["selector"])

    return {k: str(v) for k, v in paths.items()} | {"n_features": Xtr.shape[1]}

def finalize_feature_dataset(
    X_train_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    processed_dir: Path,
    final_dir: Path,
    models_dir: Path,
    tag: Optional[str] = None,
    manual_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    _resolve_repo_path(final_dir).mkdir(parents=True, exist_ok=True)

    if manual_features:
        print(f"Using manually selected feature list: ({len(manual_features)} features).")
        # Filter to only existing columns (avoid errors if user typos)
        cols = [f for f in manual_features if f in X_train_prepared.columns]
        if not cols:
            raise ValueError("None of the provided manual features exist in the training dataset.")
        # Manual mode: ignore tag entirely
        Xtr = X_train_prepared[cols].values
        Xte = X_test_prepared[cols].values

    elif tag:
        print(f"Finalizing using tag '{tag}'.")

        sel_path = _resolve_repo_path(models_dir / f"selector_{tag}.joblib")
        names_path = _resolve_repo_path(processed_dir / f"feature_names_{tag}.txt")

        if sel_path.exists():
            selector = joblib.load(sel_path)
            Xtr = selector.transform(X_train_prepared)
            Xte = selector.transform(X_test_prepared)
            cols = selector.get_feature_names_out().tolist()

        elif names_path.exists():
            cols = names_path.read_text().splitlines()
            Xtr = X_train_prepared[cols].values
            Xte = X_test_prepared[cols].values

        else:
            raise FileNotFoundError(f"Neither selector nor feature_names found for tag '{tag}'.")
    else:
        raise ValueError("You must provide either a 'manual_features' list or a 'tag'.")

    Xtr_df = pd.DataFrame(Xtr, index=X_train_prepared.index, columns=cols)
    Xte_df = pd.DataFrame(Xte, index=X_test_prepared.index,  columns=cols)

    p_Xtr = final_dir / f"X_train_final.csv"
    p_Xte = final_dir / f"X_test_final.csv"
    p_ytr = final_dir / "y_train_final.csv"
    p_yte = final_dir / "y_test_final.csv"

    save_dataframe(p_Xtr, Xtr_df)
    save_dataframe(p_Xte, Xte_df)
    save_dataframe(p_ytr, y_train)
    save_dataframe(p_yte, y_test)

    return Xtr_df, Xte_df, y_train, y_test
        


