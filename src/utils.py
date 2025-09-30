
import pandas as pd
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../hospital_readmission

def _resolve_repo_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p

def load_data(data_path: str) -> pd.DataFrame:
    path = _resolve_repo_path(data_path)
    return pd.read_csv(path)


def split_and_save_dataset(df: pd.DataFrame, target_column: str, test_size: float, output_dir: str, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

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
    train_path = f"{output_dir}/train.csv"
    test_path = f"{output_dir}/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train and test sets have been saved to:\n- {train_path}\n- {test_path}")

    return train_df, test_df
