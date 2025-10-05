# Patient Readmission Prediction

## Overview  
This project builds a reproducible machine learning pipeline to predict patient readmission using structured hospital data.

The workflow is fully automated:

- Preprocess data → clean, impute, encode, and split datasets.
- Train model → perform grid search, evaluate with cross-validation, refit the best model on full data.
- Predict & evaluate → evaluate on held-out test data with metrics, confusion matrix, and ROC/PR curves

---

## Project Structure  

```bash
.
├── config/                  
│   └── main.yaml             # Central configuration file
├── data/            
│   ├── raw/                  # Original dataset
│   ├── processed/            # Processed intermediate data
│   └── final/                # Final train/test sets (X_train_final.csv etc.)
├── models/                   # Saved models (best estimators, preprocessors)
├── notebooks/                # Jupyter notebooks for analysis
├── src/                      
│   ├── process.py               # Data preprocessing pipeline
│   ├── train_model.py           # Model training & refit on full data
│   ├── predict.py               # Model loading & evaluation on test data
│   ├── feature_selection.py     # Optional feature selection logic
│   ├── visualisation.py         # Custom plots and figures
│   └── utils.py                 # Shared helpers and path utilities
├── results/                  # Experiment results (metrics, logs)
├── pyproject.toml            # Dependencies (Poetry)
├── .gitignore
└── README.md
```
---

## Environment Setup

1. Install Poetry (https://python-poetry.org/docs/#installation)
```bash
pip install poetry
```

2. Configure Poetry to create the virtual environment inside the project (one-time setup):  
```bash
poetry config virtualenvs.in-project true
```

3. Create the environment and install dependencies:
```bash
poetry env use python
poetry install
```
If you don’t see a .venv/ folder inside your project, reset and recreate it:
```bash
poetry env remove python
poetry env use python
poetry install
```

4. Activate the virtual environment 
Windows (PowerShell):
```bash
& ".\.venv\Scripts\Activate.ps1"
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Once activated, your prompt should show (hospital_readmission) and you can run your scripts with python. 

Alternatively, you can skip activation and run scripts directly with Poetry, e.g.:
```bash
poetry run python -m src.process
```

---

## Usage

### Preprocess the data 
```bash
python -m src.process
```
- Loads raw data from data/raw/data.csv
- Cleans, imputes, encodes, and splits data
- Saves outputs to data/final/ (X_train_final.csv, y_test_final.csv, etc.)
---

### Train the model (only on training data)
```bash
python -m src.train_model
```
- Loads data from data/final/
- Performs GridSearchCV on X_train_final (with internal 5-fold CV)
- Evaluates using both train and validation metrics
- Plots ROC/PR curves for train & validation
- Refits the best model on the entire training dataset
   - Saves model → models/{model_type}_best.joblib
   - Saves metrics → results/results.json
---

### Evaluate the model on test data
```bash
python -m src.predict
```
- Loads best saved model
- Loads X_test_final.csv, y_test_final.csv
- Evaluates on held-out test data
- Computes accuracy, F1, ROC AUC, PR AUC
- Plots confusion matrix and curves (train vs test)
- Saves results to:
   - results/predictions.csv
   - results/prediction_metrics.json
---

## Configuration

All pipeline parameters are defined in one YAML file managed by Hydra:
 `config/main.yaml`.

Examples of settings you can control:
- **Preprocessing** → imputation method, scaling, feature engineering steps  
- **Model** → type (`random_forest`, `logistic_regression`), hyperparameters  
- **Cross-validation** → number of folds, scoring metric  

You can override parameters directly from the command line:
```bash
python src/train_model.py model.type=logistic_regression

```
---

## Jupyter Notebooks
Notebooks are provided for exploratory analysis and quick model testing:
- data_inspection.ipynb
- evaluate_models.ipynb

To use them with your Poetry environment:
```bash
poetry run pip install jupyter ipykernel
poetry run python -m ipykernel install --user --name=hospital-readmission
```

Then select the kernel hospital-readmission in VS Code or JupyterLab.


