import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc

def detect_outliers(df, threshold=1.5, method='iqr'):
    """
    Detect outliers in a DataFrame based on skewness values.
    
    Parameters:
    - df: DataFrame containing numerical features.
    - threshold: Skewness threshold (default is 1.5).
    - method: Method to use for outlier detection ('iqr' or 'z-score'). Default is 'iqr'.
    
    Returns:
    - outliers: Dictionary containing indices of detected outliers for each feature.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = {}
    for column in numeric_columns:
        skewness = df[column].skew()
        if abs(skewness) > threshold:
            if method == 'iqr':
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                feature_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
            elif method == 'z-score':
                z_scores = np.abs(stats.zscore(df[column]))
                feature_outliers = np.where(z_scores > threshold)[0]
            outliers[column] = feature_outliers
    return outliers

def plot_outliers(df, outliers):
    """
    Plot scatter plots of features with outliers and non-outliers.
    
    Parameters:
    - df: DataFrame containing numerical features.
    - outliers: Dictionary containing indices of outliers for each feature.
    """
    for column, outlier_indices in outliers.items():
        non_outliers = df.drop(outlier_indices)
        plt.figure(figsize=(8, 6))
        plt.scatter(non_outliers.index, non_outliers[column], color='blue', label='Non-Outliers')
        plt.scatter(outlier_indices, df.loc[outlier_indices, column], color='red', label='Outliers')
        plt.title(f'Feature: {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_roc_curve_for_classes(clf, x, y, class_labels, title):
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        # Convert the multi-class labels into binary labels for the current class
        y_binary = (y == label).astype(int)
        # Predict the probability scores for the current class
        y_score = clf.predict_proba(x)[:, i]
        # Compute the ROC curve
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        # Compute the AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve - Class {label} (AUC = {roc_auc:.2f})')
    
    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"../plots/{title}.jpg", dpi=100)
    plt.close()
