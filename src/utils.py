"""
Utility helper functions for Water Quality Anomaly Detection.
Provides authentic example sample retrieval and formatting utilities.
"""

import pandas as pd
import numpy as np

# Feature details with scientific units and typical acceptable ranges
FEATURE_METADATA = {
    'ph': {'unit': 'pH units', 'normal_range': (6.5, 8.5), 'desc': 'Acidity / Alkalinity level'},
    'Hardness': {'unit': 'mg/L', 'normal_range': (100.0, 250.0), 'desc': 'Calcium and magnesium content'},
    'Solids': {'unit': 'ppm (TDS)', 'normal_range': (5000.0, 30000.0), 'desc': 'Total Dissolved Solids'},
    'Chloramines': {'unit': 'ppm', 'normal_range': (4.0, 10.0), 'desc': 'Disinfectant compound concentration'},
    'Sulfate': {'unit': 'mg/L', 'normal_range': (200.0, 400.0), 'desc': 'Dissolved sulfate salts'},
    'Conductivity': {'unit': 'μS/cm', 'normal_range': (200.0, 600.0), 'desc': 'Electrical conductivity'},
    'Organic_carbon': {'unit': 'ppm', 'normal_range': (5.0, 20.0), 'desc': 'Total organic carbon level'},
    'Trihalomethanes': {'unit': 'μg/L', 'normal_range': (20.0, 100.0), 'desc': 'Byproduct of water chlorination'},
    'Turbidity': {'unit': 'NTU', 'normal_range': (1.0, 5.0), 'desc': 'Clarity / cloudiness measure'}
}

def get_example_observations():
    """
    Retrieves authentic NORMAL and ANOMALOUS sample observations from the dataset.
    Returns dictionaries of feature key-value pairs.
    """
    from preprocess import load_raw_dataset, build_and_apply_preprocessing, load_pipeline
    import joblib
    
    df_raw = load_raw_dataset()
    pipeline = load_pipeline()
    X_scaled, X_raw, _, _, _ = build_and_apply_preprocessing(df_raw, is_training=False, pipeline_dict=pipeline)
    
    iso_forest = joblib.load('models/isolation_forest.pkl')
    preds = iso_forest.predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    
    # Pick a typical normal sample (high decision score near median)
    normal_indices = np.where(preds == 1)[0]
    median_normal_idx = normal_indices[np.argsort(scores[normal_indices])[len(normal_indices)//2]]
    
    # Pick a distinct anomalous sample (most negative decision score)
    anom_indices = np.where(preds == -1)[0]
    extreme_anom_idx = anom_indices[np.argmin(scores[anom_indices])]
    
    # Fill any NaNs with dataset median for clean display
    normal_sample = X_raw.iloc[median_normal_idx].to_dict()
    anom_sample = X_raw.iloc[extreme_anom_idx].to_dict()
    
    for k in normal_sample:
        if pd.isna(normal_sample[k]):
            normal_sample[k] = pipeline['feature_stats'][k]['median']
        else:
            normal_sample[k] = float(normal_sample[k])
            
    for k in anom_sample:
        if pd.isna(anom_sample[k]):
            anom_sample[k] = pipeline['feature_stats'][k]['median']
        else:
            anom_sample[k] = float(anom_sample[k])
            
    return normal_sample, anom_sample
