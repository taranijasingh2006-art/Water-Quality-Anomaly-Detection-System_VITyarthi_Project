"""
Data Preprocessing Module for Water Quality Anomaly Detection.
Handles dataset loading, missing value imputation, feature scaling,
and preprocessing pipeline saving/loading.
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLUMNS = [
    'ph',
    'Hardness',
    'Solids',
    'Chloramines',
    'Sulfate',
    'Conductivity',
    'Organic_carbon',
    'Trihalomethanes',
    'Turbidity'
]

LABEL_COLUMN = 'Potability'

def load_raw_dataset(data_path='data/water_potability.csv'):
    """Load the raw CSV dataset from disk."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    return df

def build_and_apply_preprocessing(df, is_training=True, pipeline_dict=None):
    """
    Preprocess water quality dataframe.
    - Separates feature matrix X and ground-truth label y.
    - Imputes missing values using Median strategy.
    - Scales numerical features using StandardScaler.
    - Returns preprocessed X, raw X, y, and fitted pipeline dict.
    """
    stats = {}
    stats['original_shape'] = df.shape
    
    # Filter features that exist in dataframe
    present_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X_raw = df[present_features].copy()
    
    stats['missing_per_column'] = df[present_features].isnull().sum().to_dict()
    stats['total_missing_values'] = sum(stats['missing_per_column'].values())
    
    y = df[LABEL_COLUMN].copy() if LABEL_COLUMN in df.columns else None
    
    if is_training:
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_imputed = imputer.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Calculate feature baseline statistics for validation & CLI hints
        feature_stats = {}
        for idx, col in enumerate(present_features):
            col_vals = X_raw[col].dropna()
            feature_stats[col] = {
                'mean': float(col_vals.mean()),
                'std': float(col_vals.std()),
                'min': float(col_vals.min()),
                'q25': float(col_vals.quantile(0.25)),
                'median': float(col_vals.median()),
                'q75': float(col_vals.quantile(0.75)),
                'max': float(col_vals.max())
            }
            
        pipeline_dict = {
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': present_features,
            'feature_stats': feature_stats
        }
    else:
        if pipeline_dict is None:
            raise ValueError("pipeline_dict must be provided when is_training=False")
        imputer = pipeline_dict['imputer']
        scaler = pipeline_dict['scaler']
        
        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)
        
    stats['final_shape'] = X_scaled.shape
    stats['feature_names'] = present_features
    
    return X_scaled, X_raw, y, pipeline_dict, stats

def save_pipeline(pipeline_dict, save_path='models/preprocessing_pipeline.pkl'):
    """Save preprocessing pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline_dict, save_path)
    print(f"[PREPROCESS] Preprocessing pipeline saved to: {save_path}")

def load_pipeline(load_path='models/preprocessing_pipeline.pkl'):
    """Load preprocessing pipeline from disk using joblib."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Pipeline file not found at: {load_path}")
    pipeline_dict = joblib.load(load_path)
    return pipeline_dict

if __name__ == '__main__':
    df = load_raw_dataset()
    X_scaled, X_raw, y, pipeline_dict, stats = build_and_apply_preprocessing(df, is_training=True)
    print("Preprocessed X shape:", X_scaled.shape)
    print("Dataset stats:", stats)
    save_pipeline(pipeline_dict)
