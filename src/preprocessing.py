"""
Data preprocessing utilities for ML projects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    A comprehensive data preprocessing class.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant')
            columns (list): Columns to impute, if None, impute all
        
        Returns:
            pd.DataFrame: Dataset with imputed values
        """
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy=strategy)
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                self.imputers[col] = imputer
        
        return df
    
    def encode_categorical_features(self, df, method='onehot', columns=None):
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataset
            method (str): Encoding method ('onehot', 'label')
            columns (list): Columns to encode
        
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        encoded_df = df.copy()
        
        for col in columns:
            if method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Create DataFrame with encoded features
                encoded_df = pd.concat([
                    encoded_df.drop(columns=[col]),
                    pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                ], axis=1)
                
                self.encoders[col] = encoder
                
            elif method == 'label':
                encoder = LabelEncoder()
                encoded_df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
        
        return encoded_df
    
    def scale_features(self, df, method='standard', columns=None):
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataset
            method (str): Scaling method ('standard', 'minmax')
            columns (list): Columns to scale
        
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        scaled_df = df.copy()
        
        for col in columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            scaled_df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        return scaled_df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (list): Columns to check for outliers
            method (str): Outlier detection method ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Dataset without outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        clean_df = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = z_scores <= threshold
            
            clean_df = clean_df[mask]
        
        return clean_df
    
    def create_features(self, df):
        """
        Create new features from existing ones.
        Override this method for custom feature engineering.
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Dataset with new features
        """
        # Example feature engineering - customize as needed
        feature_df = df.copy()
        
        # Add polynomial features for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col.endswith('_squared') or col.endswith('_sqrt'):
                continue
            feature_df[f"{col}_squared"] = df[col] ** 2
            feature_df[f"{col}_sqrt"] = np.sqrt(np.abs(df[col]))
        
        return feature_df
    
    def preprocess(self, df, target_column=None, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target column name
            test_size (float): Test set size
            random_state (int): Random state for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) if target_column provided,
                   else preprocessed DataFrame
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create new features
        df = self.create_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale numerical features
        df = self.scale_features(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        self.feature_names = df.columns.tolist()
        if target_column:
            self.feature_names.remove(target_column)
        
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return df
