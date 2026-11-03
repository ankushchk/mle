# Case Study 1: Hospital Readmission Prediction

This case study builds a binary classification model to predict whether a patient will be readmitted to hospital. It uses logistic regression with L2 regularization and evaluates the model with ROC-AUC, an ROC curve, and a confusion matrix.

## Objective

Predict the `readmitted` outcome:

- `yes` becomes class `1`.
- `no` becomes class `0`.

The model is intended for educational analysis. It is not a clinical decision tool and should not be used to make patient-care decisions without clinical validation, governance, and fairness review.

## Dataset

`hospital_readmissions.csv` is included in this directory. It contains 25,000 rows and 17 columns:

- Patient age band: `age`
- Utilization and stay measures: `time_in_hospital`, `n_lab_procedures`, `n_procedures`, `n_medications`, `n_outpatient`, `n_inpatient`, `n_emergency`
- Clinical categories: `medical_specialty`, `diag_1`, `diag_2`, `diag_3`
- Diabetes-related indicators: `glucose_test`, `A1Ctest`, `change`, `diabetes_med`
- Target: `readmitted`

The notebook checks for missing values and duplicate rows before modeling. The categorical columns are one-hot encoded with `drop_first=True`; numeric columns are standardized using statistics learned from the training set.

## Methodology

1. Load the CSV from the current directory. In Google Colab, the notebook falls back to `/content/hospital_readmissions.csv`.
2. Inspect schema, missing values, and duplicate rows.
3. Encode the target as a binary value.
4. One-hot encode categorical predictors.
5. Split data into stratified 80% training and 20% test sets with `random_state=42`.
6. Standardize numeric predictors using `StandardScaler` fitted only on the training set.
7. Train `LogisticRegression` with L2 regularization.
8. Report ROC-AUC and visualize the ROC curve and confusion matrix.

## Requirements

Python 3.9 or newer is recommended. Install the dependencies with:

```bash
python -m pip install numpy pandas scikit-learn matplotlib jupyter
```

## Run

From this directory, open the notebook:

```bash
jupyter notebook case-study-1.ipynb
```

Alternatively, run it with JupyterLab or upload both the notebook and CSV to Google Colab. The notebook must be run from `case-study-1` when using the local relative data path.

## Evaluation

ROC-AUC evaluates ranking quality across classification thresholds and is more informative than accuracy when the positive class is not evenly distributed. The confusion matrix shows the count of true negatives, false positives, false negatives, and true positives at the model's default threshold.

The notebook does not tune the classification threshold. For a production use case, select a threshold based on the relative cost of missed readmissions and unnecessary interventions, then validate it on an untouched dataset.

## Limitations and next steps

- A single train/test split gives limited uncertainty information; cross-validation and confidence intervals would improve the analysis.
- Logistic regression assumes a linear relationship in the encoded feature space.
- The dataset's categories and target definition may not represent a current hospital population.
- Calibration, subgroup performance, missingness policy, and temporal validation are not covered.
- Potential improvements include class weighting, regularized tree models, threshold selection using a clinical cost function, probability calibration, and external validation.
