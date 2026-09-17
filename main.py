"""
Command Line Application for Water Quality Anomaly Detection.
Provides interactive options for entering water measurements,
testing dataset examples, inspecting model comparison metrics,
and performing robust anomaly predictions.
"""

import sys
import os
import joblib

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import WaterQualityPredictor
from utils import get_example_observations, FEATURE_METADATA

def print_header():
    print("\n==================================================")
    print("   AI WATER QUALITY ANOMALY DETECTION SYSTEM")
    print("==================================================")
    print(" Unsupervised Machine Learning Model (Isolation Forest)")
    print(" Evaluates multivariate water parameters for anomalies")
    print("==================================================")

def display_prediction_results(res):
    print("\n--------------------------------------------------")
    print("               PREDICTION RESULT")
    print("--------------------------------------------------")
    status_str = res['prediction']
    if status_str == 'NORMAL':
        print(f" Prediction Status   : \033[92m[ NORMAL ]\033[0m" if os.name != 'nt' else f" Prediction Status   : [ NORMAL ]")
    else:
        print(f" Prediction Status   : \033[91m[ ANOMALOUS ]\033[0m" if os.name != 'nt' else f" Prediction Status   : [ ANOMALOUS ]")
        
    print(f" Isolation Forest Score: {res['anomaly_score']:+.4f}  (Negative score = anomalous)")
    
    if res.get('lof_prediction'):
        print(f" Local Outlier Factor : [ {res['lof_prediction']} ]")
    if res.get('ocsvm_prediction'):
        print(f" One-Class SVM        : [ {res['ocsvm_prediction']} ]")
        
    print("\n Parameter Deviation Analysis:")
    if res['parameter_deviations']:
        for dev in res['parameter_deviations']:
            print(f"  [!] {dev}")
    else:
        print("  [✓] All feature values fall within multivariate normal statistical range.")
        
    print("--------------------------------------------------")
    print(" Scientific Disclaimer: An anomaly flag indicates statistical")
    print(" divergence from learned patterns, NOT immediate water toxicity.")
    print("--------------------------------------------------")

def handle_manual_input(predictor):
    print("\n--- ENTER WATER QUALITY MEASUREMENTS ---")
    print(" (Press Enter without typing to accept dataset default median)")
    
    input_sample = {}
    stats = predictor.feature_stats
    
    for feat in predictor.feature_names:
        meta = FEATURE_METADATA.get(feat, {'unit': '', 'desc': ''})
        default_val = stats[feat]['median']
        min_v = stats[feat]['min']
        max_v = stats[feat]['max']
        
        while True:
            prompt = f" Enter {feat} [{meta['desc']} ({meta['unit']}), Dataset range: {min_v:.1f} - {max_v:.1f} | Default: {default_val:.2f}]: "
            user_str = input(prompt).strip()
            
            if user_str == "":
                input_sample[feat] = float(default_val)
                break
            try:
                val = float(user_str)
                # Validation checks for physical limits
                if feat == 'ph' and (val < 0.0 or val > 14.0):
                    print("  [Error] pH must be between 0.0 and 14.0. Please re-enter.")
                    continue
                if val < 0.0:
                    print(f"  [Error] {feat} cannot be negative. Please re-enter.")
                    continue
                input_sample[feat] = val
                break
            except ValueError:
                print("  [Error] Invalid input. Please enter a numerical value or press Enter.")
                
    res = predictor.predict_observation(input_sample)
    display_prediction_results(res)

def handle_example_input(predictor, is_normal=True):
    normal_obs, anom_obs = get_example_observations()
    sample = normal_obs if is_normal else anom_obs
    label_type = "NORMAL" if is_normal else "ANOMALOUS"
    
    print(f"\n--- TESTING AUTHENTIC DATASET EXAMPLE ({label_type}) ---")
    print("Observation Feature Values:")
    for k, v in sample.items():
        unit = FEATURE_METADATA.get(k, {}).get('unit', '')
        print(f"  - {k:<18}: {v:.2f} {unit}")
        
    res = predictor.predict_observation(sample)
    display_prediction_results(res)

def display_model_summary():
    print("\n==================================================")
    print("    WATER QUALITY ANOMALY MODEL SUMMARY")
    print("==================================================")
    print(" Primary Model    : Isolation Forest (Tree Isolation)")
    print(" Benchmark Model 1: Local Outlier Factor (LOF - Density)")
    print(" Benchmark Model 2: One-Class SVM (Boundary-RBF)")
    print(" Preprocessing    : Median Imputation + StandardScaler")
    print(" Dataset Source   : Public Water Quality Dataset (3,276 samples)")
    print(" Feature Count    : 9 Numerical Features")
    print(" Contamination    : 5.0% (0.05)")
    print(" Saved Artifacts  : models/isolation_forest.pkl, preprocessing_pipeline.pkl")
    print(" Generated Graphs : results/*.png")
    print("==================================================")

def main():
    try:
        predictor = WaterQualityPredictor()
    except Exception as e:
        print(f"\n[Error] Could not initialize predictor: {e}")
        print("Please train the model first by executing: python src/train.py")
        sys.exit(1)
        
    print_header()
    
    while True:
        print("\nCHOOSE AN OPTION:")
        print(" 1. Enter water-quality values manually")
        print(" 2. Use an example NORMAL water sample (from dataset)")
        print(" 3. Use an example ANOMALOUS water sample (from dataset)")
        print(" 4. View model summary & dataset statistics")
        print(" 5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            handle_manual_input(predictor)
        elif choice == '2':
            handle_example_input(predictor, is_normal=True)
        elif choice == '3':
            handle_example_input(predictor, is_normal=False)
        elif choice == '4':
            display_model_summary()
        elif choice == '5':
            print("\nExiting system. Thank you!\n")
            break
        else:
            print("\n [Error] Invalid selection. Please enter a number between 1 and 5.")
            continue
            
        repeat = input("\nEnter another observation? (y/n): ").strip().lower()
        if repeat not in ['y', 'yes']:
            print("\nExiting system. Thank you!\n")
            break

if __name__ == '__main__':
    main()
