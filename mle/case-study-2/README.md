# Case Study 2: Credit Card Fraud Detection

This case study detects fraudulent credit-card transactions in a highly imbalanced dataset. It uses SMOTE to rebalance the training data, XGBoost for classification, threshold analysis to explore the precision-recall trade-off, and feature importance for interpretation.

## Objective

Predict the `Class` label:

- `0`: legitimate transaction
- `1`: fraudulent transaction

Fraud detection is a cost-sensitive problem. Missing a fraud and incorrectly flagging a legitimate transaction have different consequences, so accuracy alone is not an appropriate primary metric.

## Dataset

The notebook downloads `creditcard.csv` automatically when it is not already present. The source is the public TensorFlow storage mirror of the commonly used Kaggle credit-card fraud dataset:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

The dataset has 284,807 transactions, 30 predictors, and a binary `Class` target. Only 492 transactions are fraudulent, so fraud represents approximately 0.17% of the data. Most `V1` through `V28` columns are anonymized PCA components. `Time` and `Amount` retain their original meanings.

The downloaded CSV is ignored by this case study's workflow after it has been loaded; it is a generated local artifact and should not be committed.

## Methodology

1. Download or load the dataset.
2. Inspect the first rows and class distribution.
3. Separate predictors and target.
4. Split into stratified 80% training and 20% test sets with `random_state=42`.
5. Fit `StandardScaler` on the training `Time` and `Amount` columns only, then transform both splits. This prevents test-set leakage.
6. Apply SMOTE only to the training set. The untouched test set preserves the real-world class distribution.
7. Train an `XGBClassifier` on the resampled training data.
8. Generate test-set probabilities and compare classification reports and confusion matrices at thresholds `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`.
9. Plot the model's top 15 built-in feature importance values.

## Requirements

Python 3.9 or newer is recommended. Install the dependencies with:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
```

## Run

From this directory, open the notebook:

```bash
jupyter notebook case-study-2.ipynb
```

The first notebook cell installs `imbalanced-learn` and `xgboost` for Google Colab. In a local environment, installing dependencies before starting Jupyter is preferred. Internet access is required for the first run unless `creditcard.csv` has already been placed in this directory.

## Interpreting results

The threshold controls how much probability is required to label a transaction as fraud:

- Lower thresholds generally increase recall and catch more fraud, but create more false positives.
- Higher thresholds generally increase precision, but may miss more fraud.
- Compare precision, recall, F1, and the confusion matrix at each threshold instead of selecting `0.5` automatically.

The best threshold depends on operational review capacity and the business cost of false positives versus false negatives. It should be selected using a validation set or cross-validation, not chosen solely from the final test results.

## Limitations and next steps

- SMOTE can create synthetic examples that do not reflect all real fraud patterns.
- A random split may overstate performance when transactions have temporal or account-level dependence; a time-aware split is a stronger production simulation.
- The notebook does not report PR-AUC, which is especially useful for rare-positive classification.
- XGBoost's built-in feature importance is not causal explanation and can be unstable with correlated features.
- Improvements include PR-AUC and recall-at-fixed-precision reporting, cost-based threshold optimization, probability calibration, temporal validation, model monitoring, and SHAP-based explanations.
