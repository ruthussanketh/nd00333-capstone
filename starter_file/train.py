from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.metrics import roc_auc_score


data_path = "https://raw.githubusercontent.com/ruthussanketh/nd00333-capstone/master/heart_failure_clinical_records_dataset.csv"

ds = TabularDatasetFactory.from_delimited_files(path=data_path)

x = ds.to_pandas_dataframe().dropna()
y = x.pop("DEATH_EVENT")

# Splitting data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

run = Run.get_context()

def main():
    # Adding arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", np.float(AUC_weighted))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
