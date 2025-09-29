
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

from src.visualisation import inspect_data, plot_correlation_matrix, plot_class_counts, plot_outliers_per_subject, \
    plot_outliers_per_feature, plot_pca_variance, plot_pca_feature_importance
from src.utils import load_data
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import SequentialFeatureSelector
from typing import Tuple, Union, List, Optional
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import zscore



@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    print(f"Loading data using {config.data.raw}")

    df_raw = load_data(config.data.raw)
    inspect_data(df_raw)

    # Check for duplicates
    if df_raw['inpatient.number'].is_unique:
        print("No duplicates")
    else:
        print("There are duplicates")

    # Check for unbalanced classes
    plot_class_counts(df_raw, 're.admission.within.6.months')


def remove_dead_patients(df: pd.DataFrame) -> pd.DataFrame:

    # Filter patients who died within 6 months
    dead_patients = df[df['death.within.6.months'] == 1]
    df.drop(dead_patients.index, inplace=True)
    #print_readmission_status_summary(dead_patients)

    # Calculate the percentage of patients who died
    total_patients = len(df)
    dead_patients = df[df['death.within.6.months'] == 1]
    num_dead_patients = len(dead_patients)
    percentage_dead = (num_dead_patients / total_patients) * 100
    print(f"\nPercentage of patients who died within 6 months: {percentage_dead:.2f}%")

    # Dropping features related to death
    death_features = ['death.within.6.months', 'death.within.3.months', 'death.within.28.days']
    df_processed = df.drop(columns=death_features)
    print(f"\nRemoved features: {', '.join(death_features)}")

    # The feature 'DestinationDischarge' and 'outcome.during.hospitalization' still contained patients classified as
    # 'Died', so that are dropped here.
    df_processed.drop(df_processed[df_processed['DestinationDischarge'] == 'Died'].index, inplace=True)

    return df_processed


def process_cat_variables(df_cat: pd.DataFrame) -> pd.DataFrame:
    df_cat= df_cat.copy()
    df_cat.drop(
        ['admission.way', 'discharge.department', 'type.II.respiratory.failure', 'consciousness', 'oxygen.inhalation'],
        axis=1, inplace=True)

    # Combine 'workers' and 'Officer' into 'Others' in 'occupation'
    df_cat['occupation'] = df_cat['occupation'].replace(['worker'], 'farmer')
    df_cat['occupation'] = df_cat['occupation'].replace(['farmer'], 'farmer_worker')
    df_cat['occupation'] = df_cat['occupation'].replace(['Officer'], 'Others')
    # Replace 'Others' with NaN (we will infere the missing values later)
    df_cat['occupation'] = df_cat['occupation'].replace('Others', np.nan)

    # Replace 'Unknown' with NaN in 'DestinationDischarge'
    df_cat['DestinationDischarge'] = df_cat['DestinationDischarge'].replace('Unknown', pd.NA)

    # Replace 'Left' and 'Right' with 'Single' in 'type.of.heart.failure': it seems that the information is in the gravity of heart failure, not in the heart side
    df_cat['type.of.heart.failure'] = df_cat['type.of.heart.failure'].replace(['Left', 'Right'], 'Single')

    # Combine 'GeneralWard' and 'ICU' into 'Others' in 'admission.ward'
    df_cat['admission.ward'] = df_cat['admission.ward'].replace(['GeneralWard', 'ICU'], 'Others')

    df_cat_final = convert_to_ordinal_variables(df_cat)

    return df_cat_final


def process_binary_variables(df_binary: pd.DataFrame) -> pd.DataFrame:
    df_binary = df_binary.copy()
    # Group diseases and create binary variables
    df_binary['cardiovascular_disease'] = (df_binary[['myocardial.infarction', 'congestive.heart.failure',
                                                      'peripheral.vascular.disease', 'cerebrovascular.disease']].sum(
        axis=1) > 0).astype(int)
    df_binary['metabolic_chronic_disease'] = (
                df_binary[['diabetes', 'moderate.to.severe.chronic.kidney.disease']].sum(axis=1) > 0).astype(int)
    df_binary['neurological_disease'] = (df_binary[['dementia', 'hemiplegia']].sum(axis=1) > 0).astype(int)
    df_binary['cancer'] = (df_binary[['leukemia', 'malignant.lymphoma', 'solid.tumor']].sum(axis=1) > 0).astype(int)

    # Group chronic diseases
    chronic_conditions = ['Chronic.obstructive.pulmonary.disease', 'congestive.heart.failure']
    df_binary['chronic_heart_pulmonary_disease'] = (df_binary[chronic_conditions].sum(axis=1) > 0).astype(int)

    return df_binary


def process_ordinal_features(df: pd.DataFrame, ordinal_features, df_categorical: pd.DataFrame, df_numerical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Print value counts for each ordinal feature
    for feature in ordinal_features:
        print(f'{feature}:\n', df[feature].value_counts(), '\n')

    # Add ordinal features to the categorical DataFrame
    df_cat = pd.concat([df_categorical, df[ordinal_features]], axis=1)

    # Remove ordinal features from the numerical DataFrame
    df_num = df_numerical.copy()
    df_num = df_num.drop(labels=ordinal_features, axis=1, inplace=False)

    return df_cat, df_num


def check_nan_columns(df: pd.DataFrame, threshold: float = 0.25) -> None:
    pd.set_option('display.max_rows', 100)
    high_nan = df.columns[df.isna().sum() > len(df) * threshold]
    percentage = df[high_nan.tolist()].isnull().sum() * 100 / len(df)

    print(f"How many columns with %NaN > {threshold * 100}%? {high_nan.size}")
    for col, perc in percentage.sort_values().items():
        print(f"{col}: {perc:.2f}%")


def drop_high_nan_columns(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    high_nan_columns = df.columns[df.isna().sum() > len(df) * threshold]
    print(f"Dropping {len(high_nan_columns)} columns with %NaN > {threshold * 100}%:")
    print(high_nan_columns.tolist())
    return df.drop(columns=high_nan_columns)


def print_readmission_status_summary(dead_patients: pd.DataFrame) -> None:
    # Calculate readmission status counts
    dead_readmitted_count = (
        dead_patients['re.admission.within.6.months']
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    # Define readmission status messages
    status_messages = {
        0: "were NOT readmitted.",
        1: "WERE readmitted."
    }

    # Print the summary
    print("\nReadmission status of patients who died within 6 months:")
    for status, count in dead_readmitted_count.items():
        print(f"{count} patient(s) {status_messages[status]}")


def analyze_categorical_features(df: pd.DataFrame) -> None:
    print("Categorical Feature Analysis")
    print("=" * 40)

    # Check missing values for each categorical feature
    print("\nMissing Values:")
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.concat([missing_values, missing_percentage], axis=1, keys=["Count", "Percentage"])
    print(missing_info[missing_info["Count"] > 0])  # Print only features with missing values

    # Visualize value counts for each categorical feature
    print("\nValue Counts:")
    for feature in df.columns:
        value_counts = df[feature].value_counts()
        print(value_counts)
        print(f"Total Unique Values: {value_counts.count()}")
        print("-" * 40)


class OutlierToNan(BaseEstimator, TransformerMixin):
    def __init__(self, move_outliers=True, nan_threshold=0.4, outlier_rules=None):
        self.move_outliers = move_outliers
        self.nan_threshold = nan_threshold
        self.outlier_rules = outlier_rules if outlier_rules else {
            'systolic.blood.pressure': (1, np.inf),  # Replace 0 with NaN
            'diastolic.blood.pressure': (1, np.inf),  # Replace 0 with NaN
            'map': (1, np.inf),  # Replace 0 with NaN
            'height': (1, np.inf),  # Replace values < 1 with NaN
            'weight': (10, np.inf),  # Replace values < 10 with NaN
            'BMI': (4, 100),  # Replace values < 4 or > 100 with NaN
        }

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        X = X.copy()  # Ensure we don't modify the original DataFrame

        # Apply outlier rules
        if self.move_outliers:
            for column, (min_val, max_val) in self.outlier_rules.items():
                if column in X.columns:
                    X.loc[(X[column] < min_val) | (X[column] > max_val), column] = np.nan

        # Drop columns with NaN percentage above the threshold
        Feat_nan = X.columns[X.isna().mean() > self.nan_threshold]
        X = X.drop(columns=Feat_nan, errors='ignore')

        return X


def convert_to_ordinal_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Define the mapping for ordinal encoding
    mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}

    # Apply the mapping
    df['Killip.grade'] = df['Killip.grade'].map(mapping).astype(int)
    df['NYHA.cardiac.function.classification'] = df['NYHA.cardiac.function.classification'].map(mapping).astype(int)

    return df


def create_exam_feature(df_num: pd.DataFrame, df_binary: pd.DataFrame) -> pd.DataFrame:
    df_exam = pd.DataFrame()
    df_exam['exam'] = (~df_num['glucose.blood.gas'].isna()).astype('int')

    # Check if the variable is meaningful
    # plot_distributions(df_exam, y, 're.admission.within.6.months')
    df_bin_final = pd.concat([df_binary, df_exam], axis=1)

    return df_bin_final


def evaluate_imputation_methods(
        df_num_final: pd.DataFrame,
        y: pd.Series,
        model: ClassifierMixin,  # Accepts any classification model
        n_splits: int = 3):

    # Store min and max values of features (for IterativeImputer)
    arr_min = df_num_final.min()
    arr_max = df_num_final.max()

    # Simple Imputer Evaluation 
    score_simple_imputer = pd.DataFrame()
    for strategy in ("mean", "median"):
        estimator = make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy=strategy),
            model  # Use the passed model instead of a fixed RandomForestClassifier
        )
        score_simple_imputer[strategy] = cross_val_score(
            estimator, df_num_final, y, scoring="f1", cv=n_splits
        )

    # Iterative Imputer Evaluation 
    score_iterative_imputer = pd.DataFrame()
    n_features = (10, 12, 15)
    for nf in n_features:
        estimator = make_pipeline(
            IterativeImputer(
                missing_values=np.nan,
                random_state=0,
                sample_posterior=True,
                n_nearest_features=nf,
                min_value=arr_min,
                max_value=arr_max
            ),
            model,  # Use the specified model
        )
        score_iterative_imputer[nf] = cross_val_score(
            estimator, df_num_final, y, scoring="f1", cv=n_splits
        )

    # Combine and Return Results 
    scores = pd.concat(
        [score_simple_imputer, score_iterative_imputer],
        keys=["SimpleImputer", "IterativeImputer"],
        axis=1
    )

    return scores


class FeatureSelector:
    def __init__(
        self,
        estimator: BaseEstimator,
        n_features: int,
        corr_threshold: float = 0.85,
        scoring: str = 'accuracy',
        cv: int = 3,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.corr_threshold = corr_threshold
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def _correlation_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        correlated_features = set()
        correlation_matrix = dataset.corr().abs()
        mean_corr = correlation_matrix.mean(axis=0)

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > self.corr_threshold:
                    # Keep the feature that is less correlated with others
                    colname = correlation_matrix.columns[i] if mean_corr.iloc[i] > mean_corr.iloc[j] else correlation_matrix.columns[j]
                    correlated_features.add(colname)

        print(f"Correlation filter removed {len(correlated_features)} features.")
        return dataset.drop(columns=correlated_features)

    def _sequential_feature_selection(
        self, dataset: pd.DataFrame, labels: pd.Series, method: str
    ) -> Tuple[pd.DataFrame, List[str]]:

        if method == 'SBS':
            selector = SequentialFeatureSelector(
                self.estimator, n_features_to_select=self.n_features, direction='backward',
                scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
            )
        elif method == 'SFS':
            selector = SequentialFeatureSelector(
                self.estimator, n_features_to_select=self.n_features, direction='forward',
                scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
            )
        else:
            raise ValueError("Invalid method. Choose 'SFS' or 'SBS'.")

        selector.fit(dataset, labels)
        selected_features = dataset.columns[selector.get_support()]
        print(f"Sequential Feature Selection reduced features to {len(selected_features)}.")
        return dataset[selected_features], list(selected_features)

    def fit(
        self,
        dataset: pd.DataFrame,
        labels: pd.Series,
        corr_based: bool = True,
        method: str = 'SFS',
    ) -> Tuple[pd.DataFrame, List[str]]:

        if corr_based:
            print("Performing correlation-based feature filtering...")
            dataset = self._correlation_filter(dataset)

        print("Performing sequential feature selection...")
        return self._sequential_feature_selection(dataset, labels, method)


def apply_pca(df: pd.DataFrame, n_components: Optional[Union[int, None]] = None, plot_exp_variance: bool = True,
              plot_feat_importance: bool = True) -> pd.DataFrame:
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df_scaled)

    # Create a DataFrame with the principal components
    num_components = pcs.shape[1]
    df_pca = pd.DataFrame(data=pcs, columns=[f"PC{i}" for i in range(1, num_components + 1)])

    if plot_exp_variance:
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        plot_pca_variance(explained_variance_ratio, cumulative_variance, variance_threshold=0.95)

    if plot_feat_importance:
        # Plot feature importance
        plot_pca_feature_importance(df, pca)

    return df_pca


def analyze_outliers(df: pd.DataFrame, plot_per_subject: bool = True, plot_per_feature: bool = True) -> pd.DataFrame:
    # List features
    feature_list = df.columns.tolist()
    print(f'Train subjects before outlier removal: {df.shape}')

    # Apply Z-score normalization
    df_zscored = df.apply(zscore, axis=0)
    abs_zscores = abs(df_zscored)

    # Plot outliers
    plot_outliers_per_subject(abs_zscores) if plot_per_subject else None
    plot_outliers_per_feature(abs_zscores) if plot_per_feature else None

    return df_zscored



if __name__ == "__main__":
    main()

    #df = load_data('../config/process/process1.yaml')
    #df = inspect_data(df)

