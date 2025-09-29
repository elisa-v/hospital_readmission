
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


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
