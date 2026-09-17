"""
Unit and Integration Tests for Water Quality Anomaly Detection System.
Tests preprocessing, model inference, dataset example loading, and predictor outputs.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import load_raw_dataset, build_and_apply_preprocessing
from predict import WaterQualityPredictor
from utils import get_example_observations, FEATURE_METADATA

class TestWaterQualityPipeline(unittest.TestCase):
    """Test suite for preprocessing, predictor, and utility functions."""
    
    @classmethod
    def setUpClass(cls):
        """Ensure dataset exists and load raw data."""
        cls.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/water_potability.csv'))
        cls.assertTrue(os.path.exists(cls.data_path), f"Dataset missing at {cls.data_path}")
        cls.df_raw = load_raw_dataset(cls.data_path)
        cls.predictor = WaterQualityPredictor(models_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

    def test_01_dataset_loading(self):
        """Verify dataset row count and feature columns."""
        self.assertEqual(self.df_raw.shape[0], 3276)
        self.assertIn('ph', self.df_raw.columns)
        self.assertIn('Turbidity', self.df_raw.columns)

    def test_02_preprocessing_pipeline(self):
        """Test missing value imputation and scaling."""
        X_scaled, X_raw, y, pipeline_dict, stats = build_and_apply_preprocessing(self.df_raw, is_training=True)
        self.assertEqual(X_scaled.shape[1], 9)
        self.assertFalse(np.isnan(X_scaled).any(), "Imputed scaled matrix must not contain NaNs")

    def test_03_prediction_normal_sample(self):
        """Test inference on a normal water sample."""
        sample = {
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
        res = self.predictor.predict_observation(sample)
        self.assertIn(res['prediction'], ['NORMAL', 'ANOMALOUS'])
        self.assertIsInstance(res['anomaly_score'], float)

    def test_04_prediction_extreme_anomalous_sample(self):
        """Test inference on an extreme outlier sample."""
        extreme_sample = {
            'ph': 13.8,             # Extreme pH
            'Hardness': 320.0,
            'Solids': 60000.0,      # Extreme TDS
            'Chloramines': 12.5,
            'Sulfate': 480.0,
            'Conductivity': 750.0,
            'Organic_carbon': 28.0,
            'Trihalomethanes': 120.0,
            'Turbidity': 6.5
        }
        res = self.predictor.predict_observation(extreme_sample)
        self.assertEqual(res['prediction'], 'ANOMALOUS')
        self.assertLess(res['anomaly_score'], 0.0, "Extreme outlier must yield negative score")
        self.assertTrue(len(res['parameter_deviations']) > 0, "Extreme values must trigger parameter deviation warnings")

    def test_05_example_observation_retrieval(self):
        """Test retrieval of authentic dataset examples."""
        normal_obs, anom_obs = get_example_observations()
        self.assertEqual(len(normal_obs), 9)
        self.assertEqual(len(anom_obs), 9)

if __name__ == '__main__':
    unittest.main()
