"""
Prediction Module for Water Quality Anomaly Detection.
Loads saved models and preprocessing pipeline to predict anomaly status
and compute decision scores for new water quality measurements.
"""

import os
import joblib
import pandas as pd
import numpy as np

class WaterQualityPredictor:
    """Predictor class encapsulating saved pipeline and anomaly models."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.pkl')
        self.if_path = os.path.join(models_dir, 'isolation_forest.pkl')
        self.lof_path = os.path.join(models_dir, 'lof_model.pkl')
        self.ocsvm_path = os.path.join(models_dir, 'ocsvm_model.pkl')
        
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load fitted pipeline and trained models."""
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Missing pipeline artifact at: {self.pipeline_path}. Run training first.")
        if not os.path.exists(self.if_path):
            raise FileNotFoundError(f"Missing model artifact at: {self.if_path}. Run training first.")
            
        self.pipeline = joblib.load(self.pipeline_path)
        self.iso_forest = joblib.load(self.if_path)
        
        self.lof = joblib.load(self.lof_path) if os.path.exists(self.lof_path) else None
        self.ocsvm = joblib.load(self.ocsvm_path) if os.path.exists(self.ocsvm_path) else None
        
        self.feature_names = self.pipeline['feature_names']
        self.feature_stats = self.pipeline['feature_stats']
        
    def predict_observation(self, sample_dict):
        """
        Predict anomaly status for a single observation dictionary.
        Returns prediction status, anomaly score, and parameter deviation analysis.
        """
        # Ensure input dictionary has all required feature keys
        row = []
        for feat in self.feature_names:
            val = sample_dict.get(feat, None)
            if val is None or pd.isna(val):
                # Fallback to dataset median if missing
                val = self.feature_stats[feat]['median']
            row.append(float(val))
            
        df_single = pd.DataFrame([row], columns=self.feature_names)
        
        # Preprocess using fitted pipeline
        X_imputed = self.pipeline['imputer'].transform(df_single)
        X_scaled = self.pipeline['scaler'].transform(X_imputed)
        
        # Predict with Isolation Forest
        if_pred = self.iso_forest.predict(X_scaled)[0]
        if_score = float(self.iso_forest.decision_function(X_scaled)[0])
        
        prediction_label = 'NORMAL' if if_pred == 1 else 'ANOMALOUS'
        
        # Comparison models
        lof_pred_label = None
        if self.lof is not None:
            lof_p = self.lof.predict(X_scaled)[0]
            lof_pred_label = 'NORMAL' if lof_p == 1 else 'ANOMALOUS'
            
        ocsvm_pred_label = None
        if self.ocsvm is not None:
            oc_p = self.ocsvm.predict(X_scaled)[0]
            ocsvm_pred_label = 'NORMAL' if oc_p == 1 else 'ANOMALOUS'
            
        # Parameter-level deviation detection (check if value is outside q25 - 1.5*IQR or q75 + 1.5*IQR)
        deviations = []
        for feat, val in sample_dict.items():
            if feat in self.feature_stats:
                st = self.feature_stats[feat]
                iqr = st['q75'] - st['q25']
                lower_bound = st['q25'] - 1.5 * iqr
                upper_bound = st['q75'] + 1.5 * iqr
                
                if val < lower_bound:
                    deviations.append(f"{feat}: {val:.2f} (Significantly LOWER than dataset normal range [{st['min']:.1f} - {st['max']:.1f}])")
                elif val > upper_bound:
                    deviations.append(f"{feat}: {val:.2f} (Significantly HIGHER than dataset normal range [{st['min']:.1f} - {st['max']:.1f}])")
                    
        return {
            'prediction': prediction_label,
            'anomaly_score': if_score,
            'lof_prediction': lof_pred_label,
            'ocsvm_prediction': ocsvm_pred_label,
            'parameter_deviations': deviations,
            'input_features': sample_dict
        }

if __name__ == '__main__':
    predictor = WaterQualityPredictor()
    # Test with standard typical normal sample
    test_sample = {
        'ph': 7.2,
        'Hardness': 195.0,
        'Solids': 20000.0,
        'Chloramines': 7.1,
        'Sulfate': 330.0,
        'Conductivity': 420.0,
        'Organic_carbon': 14.0,
        'Trihalomethanes': 65.0,
        'Turbidity': 3.9
    }
    res = predictor.predict_observation(test_sample)
    print("Test Prediction Output:")
    print(res)
