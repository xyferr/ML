"""
ML Workspace Utilities

This module contains common utility functions for machine learning projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath, **kwargs):
    """
    Load data from various file formats.
    
    Args:
        filepath (str): Path to the data file
        **kwargs: Additional arguments for pandas read functions
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    file_ext = filepath.split('.')[-1].lower()
    
    if file_ext == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif file_ext in ['xlsx', 'xls']:
        return pd.read_excel(filepath, **kwargs)
    elif file_ext == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif file_ext == 'json':
        return pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def basic_eda(df):
    """
    Perform basic exploratory data analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    
    Returns:
        dict: Summary statistics and info
    """
    print("Dataset Shape:", df.shape)
    print("\nColumn Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())
    
    return {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'statistics': df.describe()
    }


def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot correlation matrix for numerical columns.
    
    Args:
        df (pd.DataFrame): Dataset
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def save_model(model, filepath):
    """
    Save a trained model.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a saved model.
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def evaluate_classification_model(y_true, y_pred, target_names=None):
    """
    Evaluate a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of target class names
    
    Returns:
        dict: Evaluation metrics
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {'classification_report': report, 'confusion_matrix': cm}
