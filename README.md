# Patient Readmission Prediction

## Overview  
This project builds a machine learning pipeline to predict patient readmission using structured hospital data.  

The main goals are to:  
- Explore and preprocess patient data.  
- Train baseline models (Logistic Regression, Random Forest).  
- Evaluate performance using cross-validation.  
- Save results and trained models in a reproducible way.  

The project follows a clean ML project structure with configuration files for preprocessing, model training, and experiment tracking.

---

## Project Structure  

```bash
.
├── config/                  
│   └── main.yaml             # Central configuration file
├── data/            
│   ├── raw/                  # Original dataset
│   ├── processed/            # Cleaned dataset (train/test split)
│   └── final/                # Final predictions
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks for EDA & experiments
├── src/                      # Source code
│   ├── process.py            # Data preprocessing pipeline
│   ├── train_model.py        # Model training & evaluation
│   └── utils.py              # Helper functions
├── tests/                    # Tests
│   ├── test_process.py
│   └── test_train_model.py
├── results/                  # Experiment results (metrics, logs)
├── pyproject.toml            # Dependencies (Poetry)
├── .gitignore
└── README.md
```
---

## Environment Setup

1. Install Poetry: https://python-poetry.org/docs/#installation

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

### Train and evaluate the model 
```bash
python -m src.train_model
```
Trained models will be stored in models/ and results in results/.
---

## Configuration

Default parameters are stored in `config/main.yaml`.

Examples of settings you can control:
- **Preprocessing** → imputation method, scaling, feature engineering steps  
- **Model** → type (`random_forest`, `logistic_regression`), hyperparameters  
- **Cross-validation** → number of folds, scoring metric  

Override configuration from the command line:
```bash
python src/train_model.py model.type=logistic_regression

```
---

## Testing
Run tests with:
```bash
pytest tests/
```
