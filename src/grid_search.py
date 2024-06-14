from sklearn.model_selection import GridSearchCV

import argparse
import logging
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
import config
import model_dispatcher
import warnings
from helper import plot_roc_curve_for_classes
warnings.filterwarnings('ignore')



# Define the parameter grid
param_grid = {
    'lr': {
        'max_iter': [100, 200, 500],
        'class_weight': [{0: 1, 1: 2, 2: 2, 3: 3, 4: 2}, 'balanced', None]
    },
    'dt': {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}
train_scores = []
test_scores = []
def baseline_training(fold, model_name, scale=False, metric='roc_auc', plot_roc=False, param_grid= param_grid):
    print("#" * 20)
    print(f"### FOLD {fold} ###")
    logging.info("BASELINE TRAINING STARTED")
    logging.info(f"### FOLD {fold} ###")

    df = pd.read_csv(config.SMOTE_10FOLD_FILE, na_values="?")
    df.dropna(axis=0, inplace=True)

    df_train = df[df.skfold != fold].reset_index(drop=True)
    df_valid = df[df.skfold == fold].reset_index(drop=True)

    xtrain = df_train.drop('class', axis=1).values
    ytrain = df_train['class'].values

    xvalid = df_valid.drop('class', axis=1).values
    yvalid = df_valid['class'].values

    if scale:
        quantile_transformer = RobustScaler()
        xtrain = quantile_transformer.fit_transform(xtrain)
        xvalid = quantile_transformer.transform(xvalid)

    model = model_dispatcher.models[model_name]
    logging.info(f"Model Name: {model}")
    logging.info(f"Hyperparameters: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")

    # Grid Search for hyperparameter tuning
    if model_name in param_grid:
        grid = GridSearchCV(model, param_grid[model_name], cv=3, scoring=metric)
        grid.fit(xtrain, ytrain)
        model = grid.best_estimator_
        logging.info(f"Best parameters found: {grid.best_params_}")
    else:
        model.fit(xtrain, ytrain)

    if hasattr(model, 'predict_proba'):
        train_preds_prob = model.predict_proba(xtrain)
        test_preds_prob = model.predict_proba(xvalid)
    else:
        train_preds_prob = model.predict(xtrain)
        test_preds_prob = model.predict(xvalid)

    if metric == 'f1_score':
        train_score = metrics.f1_score(ytrain, model.predict(xtrain), average='weighted')
        test_score = metrics.f1_score(yvalid, model.predict(xvalid), average='weighted')
    elif metric == 'roc_auc':
        train_score = metrics.roc_auc_score(ytrain, train_preds_prob, multi_class='ovr')
        test_score = metrics.roc_auc_score(yvalid, test_preds_prob, multi_class='ovr')
    elif metric == 'precision':
        train_score = metrics.precision_score(ytrain, model.predict(xtrain), average='weighted')
        test_score = metrics.precision_score(yvalid, model.predict(xvalid), average='weighted')
    elif metric == 'recall':
        train_score = metrics.recall_score(ytrain, model.predict(xtrain), average='weighted')
        test_score = metrics.recall_score(yvalid, model.predict(xvalid), average='weighted')
    elif metric == 'accuracy':
        train_score = metrics.accuracy_score(ytrain, model.predict(xtrain))
        test_score = metrics.accuracy_score(yvalid, model.predict(xvalid))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    print(f"Training {metric}: {train_score}")
    print(f"Testing {metric}: {test_score}")

    logging.info(f"Training {metric}: {train_score}")
    logging.info(f"Testing {metric}: {test_score}")

    train_classification_report = metrics.classification_report(ytrain, model.predict(xtrain))
    logging.info(f"Classification Report - Training (Fold {fold}):\n{train_classification_report}")

    test_classification_report = metrics.classification_report(yvalid, model.predict(xvalid))
    logging.info(f"Classification Report - Testing (Fold {fold}):\n{test_classification_report}")
    logging.info("#" * 20)

    train_scores.append(train_score)
    test_scores.append(test_score)

    if plot_roc:
        plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1, 2, 3, 4], f'Training ROC Curve for Fold {fold + 1} using {model_name}')
        plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1, 2, 3, 4], f'Testing ROC Curve for Fold {fold + 1} using {model_name}')

    return train_score, test_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "dt", "lasso", "elastic_net", "nb"])
    parser.add_argument("--logs", type=str, default=None)
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--metric", type=str, default='roc_auc', choices=['f1_score', 'roc_auc', 'precision', 'recall', 'accuracy', 'f1_weighted'])
    parser.add_argument("--plot_roc", type=bool, default=False)
    args = parser.parse_args()

    logging.basicConfig(filename=f"{config.LOGGING_FILS}/{args.logs}.log", level=logging.INFO)
    train_roc_auc_avg = []
    test_roc_auc_avg = []

    for fold in range(args.fold):
        train_roc_auc, test_roc_auc = baseline_training(fold, args.model, args.scale, args.metric, args.plot_roc, param_grid)
        train_roc_auc_avg.append(train_roc_auc)
        test_roc_auc_avg.append(test_roc_auc)

    logging.info(f"Overall {args.metric} on all {args.fold} FOLDS in TRAINING SET={np.mean(train_roc_auc_avg)}, TESTING SET={np.mean(test_roc_auc_avg)}")
    print(f"Overall Training {args.metric}: {np.mean(train_roc_auc_avg)}, Testing {args.metric} : {np.mean(test_roc_auc_avg)}")
