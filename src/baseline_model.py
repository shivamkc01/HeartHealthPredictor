
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


train_scores = []
test_scores = []
def baseline_training(fold, model, scale=False, metric='roc_auc', plot_roc=False):
    """
    Trains a baseline model and evaluates its performance on a specific fold.

    Args:
        fold (int): Fold number for cross-validation.
        model (str): Name of the model to use (e.g., "lr" for Logistic Regression).
        scale (bool, optional): Whether to apply scaling to the training data. Defaults to False.
        metric (str, optional): Evaluation metric to use. Defaults to "roc_auc".

    Raises:
        ValueError: If the provided metric is not supported.
    """

    print("#" * 20)
    print(f"### FOLD {fold} ###")
    logging.info("BASELINE TRAINING STARTED")
    logging.info(f"### FOLD {fold} ###")

    df = pd.read_csv(config.SMOTE_10FOLD_FILE, na_values="?")
    df.dropna(axis=0, inplace=True)

    df_train = df[df.skfold != fold].reset_index(drop=True)  # Drop unnecessary index reset
    df_valid = df[df.skfold == fold].reset_index(drop=True)  # Drop unnecessary index reset

    xtrain = df_train.drop('class', axis=1).values
    ytrain = df_train['class'].values

    xvalid = df_valid.drop('class', axis=1).values
    yvalid = df_valid['class'].values

    if scale:
        quantile_transformer = RobustScaler()
        xtrain = quantile_transformer.fit_transform(xtrain)
        xvalid = quantile_transformer.transform(xvalid)

    model = model_dispatcher.models[model]
    logging.info(f"Model Name: {model}")
    logging.info(f"Hyperparameters: {model.get_params()}")

    model.fit(xtrain, ytrain)

    # Check if model predicts probabilities or labels
    if hasattr(model, 'predict_proba'):
        train_preds_prob = model.predict_proba(xtrain)
        test_preds_prob = model.predict_proba(xvalid)
    else:
        train_preds_prob = model.predict(xtrain)
        test_preds_prob = model.predict(xvalid)

    if metric == 'f1_score':
        train_score = metrics.f1_score(ytrain, model.predict(xtrain), average='weighted')
        test_score = metrics.f1_score(yvalid, model.predict(xvalid),average='weighted')
    elif metric == 'roc_auc':
        train_score = metrics.roc_auc_score(ytrain, train_preds_prob, multi_class='ovr')
        test_score = metrics.roc_auc_score(yvalid, test_preds_prob, multi_class='ovr')
    elif metric == 'precision':
        train_score = metrics.precision_score(ytrain, model.predict(xtrain),average='weighted')
        test_score = metrics.precision_score(yvalid, model.predict(xvalid),average='weighted')
    elif metric == 'recall':
        train_score = metrics.recall_score(ytrain, model.predict(xtrain),average='weighted')
        test_score = metrics.recall_score(yvalid, model.predict(xvalid),average='weighted')
    elif metric == 'accuracy':
        train_score = metrics.accuracy_score(ytrain, model.predict(xtrain))
        test_score = metrics.accuracy_score(yvalid, model.predict(xvalid))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    print(f"Training {metric}: {train_score}")
    print(f"Testing {metric}: {test_score}")

    logging.info(f"Training {metric}: {train_score}")
    logging.info(f"Testing {metric}: {test_score}")

    train_classification_report = metrics.classification_report(
        ytrain,
        model.predict(xtrain),
    )
    logging.info(f"Classification Report - Traning (Fold {fold}):\n{train_classification_report}")

    test_classification_report = metrics.classification_report(
        yvalid,
        model.predict(xvalid),
    )
    logging.info(f"Classification Report - Testing (Fold {fold}):\n{test_classification_report}")
    logging.info("#"*20)
    train_scores.append(train_score)
    test_scores.append(test_score)

    if plot_roc:
        plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1, 2, 3, 4], f' Training ROC Curve for Fold {fold+1} using {args.model} ')
        plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1, 2, 3, 4], f'Testing ROC Curve for Fold {fold+1} using {args.model} ')

    return train_score, test_score

if __name__ == "__main__":
  # Lets define argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--fold", type=int, default=5)
  parser.add_argument("--model", type=str, default="lr", choices=["lr", "dt", "nb"])
  parser.add_argument("--logs", type=str, default=None)
  parser.add_argument("--scale", type=bool, default=False)
  parser.add_argument("--metric", type=str, default='roc_auc', choices=['f1_score', 'roc_auc', 'precision', 'recall', 'accuracy', 'f1_weighted'])
  parser.add_argument("--plot_roc", type=bool, default=False)
  args = parser.parse_args()

  logging.basicConfig(filename=f"{config.LOGGING_FILS}/{args.logs}.log", level=logging.INFO)
  train_roc_auc_avg = []
  test_roc_auc_avg = []


  for fold in range(args.fold):
    train_roc_auc, test_roc_auc= baseline_training(fold, args.model, args.scale, args.metric, args.plot_roc)
    train_roc_auc_avg.append(train_roc_auc)
    test_roc_auc_avg.append(test_roc_auc)

  logging.info(f"Overall {args.metric} on all 10 FOLDs in TRAINING SET={np.mean(train_roc_auc_avg)}, TESTING SET={np.mean(test_roc_auc_avg)}") 
  print(f"Overall Training {args.metric}: {np.mean(train_roc_auc_avg)}, Testing {args.metric} : {np.mean(test_roc_auc_avg)}")
  