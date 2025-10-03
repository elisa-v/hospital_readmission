

import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

from sklearn.decomposition import PCA


def inspect_data(df: pd.DataFrame) -> None:
    print(f"Number of rows and columns: {df.shape}")
    print(df.info())
    print(df.describe())


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


def plot_histogram(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_class_counts(df: pd.DataFrame, column: str, style: str = 'ggplot') -> None:
    print(df.groupby(column).size())
    plt.style.use(style)
    sns.countplot(
        y=df['re.admission.within.6.months'],
        data=df,
        hue=df['re.admission.within.6.months'],
        palette=['lightblue', 'red'],
        legend=False
    )
    plt.ylabel("Classes")
    plt.xlabel("Count")
    plt.title(f"Class Counts for {column}")
    plt.show()


def plot_distributions(categorical_df: pd.DataFrame, target_df: pd.Series, target_col: str) -> None:

    # Ensure all categorical variables are strings
    categorical_df = categorical_df.copy()  # Explicitly copy to avoid SettingWithCopyWarning
    for col in categorical_df.columns:
        categorical_df[col] = categorical_df[col].astype('str')

    # Combine categorical DataFrame with the target column for easier splitting
    combined_df = categorical_df.copy()
    combined_df[target_col] = target_df.copy()  # Ensure a copy of target_df is used

    # Split data into the two target classes
    df_0 = combined_df[combined_df[target_col] == 0].copy()
    df_1 = combined_df[combined_df[target_col] == 1].copy()

    # Create subplots for the categorical variables
    num_vars = len(categorical_df.columns)
    ncols = 4
    nrows = (num_vars // ncols) + (1 if num_vars % ncols else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 20))
    axes = axes.flatten()  # Flatten to easily iterate over

    # Plot histograms for each variable
    for i, col in enumerate(categorical_df.columns):
        plt.sca(axes[i])  # Set the current Axes
        plt.hist([df_0[col], df_1[col]], density=True, label=['Non-readmission', 'Readmission'], alpha=0.7)
        plt.title(col)
        plt.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_binary_distributions(binary_df: pd.DataFrame, target_df: pd.Series, target_col: str) -> None:
    # Combine binary DataFrame with the target column for easier splitting
    combined_df = binary_df.copy()
    combined_df[target_col] = target_df

    # Split data into the two target classes
    df_0 = combined_df[combined_df[target_col] == 0]
    df_1 = combined_df[combined_df[target_col] == 1]

    # Create subplots for the binary variables
    num_vars = len(binary_df.columns)
    ncols = 4
    nrows = (num_vars // ncols) + (1 if num_vars % ncols else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    axes = axes.flatten()  # Flatten to easily iterate over

    # Plot histograms for each binary variable
    for i, col in enumerate(binary_df.columns):

        df_0_temp = df_0.copy()
        df_1_temp = df_1.copy()

        df_0_temp.drop(df_0_temp[df_0_temp[col] == 0].index, inplace=True)
        df_1_temp.drop(df_1_temp[df_1_temp[col] == 0].index, inplace=True)

        labels = []
        if not df_0_temp.empty:
            labels.append('Non-readmission')
        if not df_1_temp.empty:
            labels.append('Readmission')

        plt.sca(axes[i])  # Set the current Axes
        plt.hist([df_0_temp[col], df_1_temp[col]], density=True, label=labels, alpha=0.7)
        plt.title(col)
        plt.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_num_var_histograms(df: pd.DataFrame, bins: int = 20, max_per_fig: int = 20, figsize: tuple = (15, 10)) -> None:

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_groups = math.ceil(len(num_cols) / max_per_fig)

    for i in range(num_groups):
        cols_subset = num_cols[i * max_per_fig: (i + 1) * max_per_fig]
        df[cols_subset].hist(figsize=figsize, bins=bins)
        plt.tight_layout()
        plt.show()


def plot_boxplots(df: pd.DataFrame, columns: list, ncols: int = 5, figsize: tuple = (20, 20)) -> None:

    nrows = (len(columns) + ncols - 1) // ncols  # Calculate number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout(pad=5)

    # Flatten axes array to make indexing easier
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(y=col, data=df, orient='v', ax=axes[i])
        axes[i].set_title(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()


def plot_outliers_per_subject(abs_zscores: pd.DataFrame):
    # Number of outliers per subject
    bool_zscores = (abs_zscores > 3).sum(axis=1)
    bool_zscores.hist()
    plt.title("Number of outliers per subject")
    plt.show()


def plot_outliers_per_feature(abs_zscores):
    bool_zscores = (abs_zscores > 3).sum(axis=0)

    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_list_new = []
    bool_zscores_new = []

    for i in range(len(bool_zscores)):
        if bool_zscores.iloc[i] > 0:  # Use `.iloc` for positional indexing
            bool_zscores_new.append(bool_zscores.iloc[i])
            feature_list_new.append(bool_zscores.index[i])  # Access index for feature name

    ax.bar(feature_list_new, bool_zscores_new, color="red")
    ax.set_title("Number of outliers per feature")
    ax.set_ylabel("Number of Outliers")
    ax.set_xlabel("Features")
    ax.tick_params(axis='x', rotation=90, labelsize=8)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

    return feature_list_new

def plot_pairplot(df: pd.DataFrame, features: Optional[List[str]] = None, hue: str = 're.admission.within.6.months',
                  max_features: int = 20) -> None:

    # Auto-detect numerical features if not provided
    features = features or df.select_dtypes(include=['number']).columns.tolist()

    # Ensure hue column exists
    if hue not in df.columns:
        raise ValueError(f"'{hue}' column not found in the dataframe.")

    num_features = len(features)

    # If there's only one feature, pairplot is not possible
    if num_features < 2:
        print("Not enough features for pairplot. At least two numerical features are required.")
        return

    # Generate pairplots in chunks
    for i in range(0, num_features, max_features):
        subset_features = features[i:i + max_features]

        # Ensure hue is included and avoid duplicates
        selected_columns = list(dict.fromkeys(subset_features + [hue]))

        print(f"Plotting features {i + 1} to {min(i + max_features, num_features)} out of {num_features}")

        sns.pairplot(df[selected_columns], hue=hue, palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.show()


def plot_imputation_results(scores: pd.DataFrame, title: str = "Results of Different Imputation Methods"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate mean and standard deviation for error bars
    means, errors = scores.mean(), scores.std()

    # Plot horizontal bar chart with error bars
    means.plot.barh(xerr=errors, ax=ax, capsize=5, color="skyblue", edgecolor="black")

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("F1 Score")
    ax.set_yticks(np.arange(means.shape[0]))
    ax.set_yticklabels([
        "Single Imputer (Mean)", "Single Imputer (Median)",
        "Iterative Imputer (10 Nearest Features)", "Iterative Imputer (12 Nearest Features)",
        "Iterative Imputer (15 Nearest Features)"
    ])

    # Layout adjustment
    plt.tight_layout(pad=1)
    plt.show()


def plot_pca_variance(explained_variance_ratio: np.ndarray, cumulative_variance: np.ndarray, variance_threshold: float = 0.9) -> None:
    # Create figure and primary axis (Explained Variance)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for explained variance (left y-axis)
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7,
            label='Explained Variance', color='C0')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    # Secondary y-axis for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--',
             color='r', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a horizontal dashed line at the specified cumulative variance threshold
    ax2.axhline(y=variance_threshold, color='gray', linestyle='dashed', linewidth=1.5, label=f'{variance_threshold*100:.0f}% Variance Threshold')

    # Find the number of components needed to reach the threshold
    num_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1
    ax2.axvline(x=num_components_needed, color='gray', linestyle='dashed', linewidth=1.5)
    print(f"Number of components needed to reach {variance_threshold*100:.0f}% variance: {num_components_needed}")

    # Annotate the number of components needed
    ax2.text(num_components_needed + 1, variance_threshold - 0.05, f'{num_components_needed} PCs', color='gray', fontsize=12)

    # Title and legend
    fig.suptitle('PCA: Explained Variance per Component & Cumulative Variance')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.grid()
    plt.show()



def plot_pca_feature_importance(df: pd.DataFrame, pca: PCA, num_pcs: int = 5):
    # Get PCA loadings (component coefficients)
    loadings = pd.DataFrame(
        pca.components_[:num_pcs].T,  # Transpose so features are rows
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(num_pcs)]
    )

    # # Bar Plot: Feature Importance in Each PC
    # fig, axes = plt.subplots(num_pcs, 1, figsize=(10, 4 * num_pcs), sharex=True)
    # for i, ax in enumerate(axes):
    #     pc_name = f"PC{i+1}"
    #     sorted_loadings = loadings[pc_name].abs().sort_values(ascending=False)  # Sort by absolute impact
    #     sorted_loadings.plot(kind="bar", ax=ax, color="C0", alpha=0.8)
    #     ax.set_title(f"Feature Importance in {pc_name}")
    #     ax.set_ylabel("Loading Magnitude")
    #     ax.grid()
    #
    # plt.xlabel("Features")
    # plt.xticks(rotation=90)
    # plt.show()

    # Heatmap of Feature Contributions
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Contributions to Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Features")
    plt.show()
