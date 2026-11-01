# Customer Churn Prediction

This project explores customer churn in a telecom dataset and compares three
classification models for predicting whether a customer will leave.

## Project files

- `Telecom_Dataset.csv` — customer-level data used by the analysis.
- `notebook.ipynb` — data exploration, feature preparation, model training, and
  evaluation.

## Dataset

The dataset contains 7,043 customer records and 21 columns. `Churn` is the
target, with `Yes` indicating a customer who churned and `No` indicating a
customer who stayed. The remaining columns describe customer demographics,
services, contract and billing details, tenure, and charges. `customerID` is
removed before model training.

The target is imbalanced: 1,869 customers churned and 5,174 did not. The
notebook uses a stratified train/test split to preserve the class proportions.

## Analysis and modeling

The notebook:

1. Loads the CSV with pandas and inspects the data.
2. Explores churn distribution and its relationship with tenure, monthly
   charges, contract type, and total charges using plots.
3. Encodes the target as 0/1 and one-hot encodes categorical features.
4. Splits the data into training and test sets (80/20, random state 42,
   stratified by churn).
5. Trains and evaluates Logistic Regression, Random Forest, and XGBoost
   classifiers using accuracy, classification reports, and confusion matrices.

## Results

The notebook's recorded test-set results are:

| Model | Accuracy | Churn precision | Churn recall | Churn F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.42% | 0.63 | 0.54 | 0.58 |
| Random Forest | 79.91% | 0.66 | 0.50 | 0.57 |
| XGBoost | 79.21% | 0.63 | 0.53 | 0.58 |

These are the results stored in the notebook and may vary if the data or
environment changes. Accuracy alone does not capture performance on the
minority churn class, so consider churn precision and recall when comparing
models.

## Requirements

The notebook uses Python and these packages:

- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- Jupyter Notebook

Install the packages with pip:

```bash
python -m pip install pandas matplotlib seaborn scikit-learn xgboost notebook
```

## Run the project

1. Open a terminal in this project folder.
2. Install the requirements above.
3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `notebook.ipynb` and run the cells from top to bottom. Keep
   `Telecom_Dataset.csv` in the same folder so the notebook can load it using
   its relative path.
