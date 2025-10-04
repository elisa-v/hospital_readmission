
import os
from typing import Any, Union
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../hospital_readmission

def load_data(data_path: str) -> pd.DataFrame:
    path = _resolve_repo_path(data_path)
    return pd.read_csv(path)

def _resolve_repo_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p

def split_and_save_dataset(df: pd.DataFrame, target_column: str, test_size: float, output_dir: str, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    path = _resolve_repo_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Split features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Reunify features and target into DataFrames
    train_df = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

    # Save the train and test sets to CSV files
    train_path = f"{path}/train.csv"
    test_path = f"{path}/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train and test sets have been saved to:\n- {train_path}\n- {test_path}")

    return train_df, test_df

def save_final_datasets(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    output_dir: Union[str, Path]
    ) -> None:

    # Resolve absolute path relative to project root
    output_path = _resolve_repo_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define file paths
    X_train_path = output_path / "X_train_final.csv"
    X_test_path = output_path / "X_test_final.csv"
    y_train_path = output_path / "y_train_final.csv"
    y_test_path = output_path / "y_test_final.csv"

    # Save datasets
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"Final datasets saved to {output_path}")
    print(f"  ├─ X_train_final.csv: {X_train.shape}")
    print(f"  ├─ X_test_final.csv:  {X_test.shape}")
    print(f"  ├─ y_train_final.csv: {y_train.shape}")
    print(f"  └─ y_test_final.csv:  {y_test.shape}")

def save_preprocessor(preprocessor: Any, output_path: Union[str, Path] = "../models/preprocessor.joblib") -> None:
    path = _resolve_repo_path(output_path)
    os.makedirs(path.parent, exist_ok=True)

    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")
    
