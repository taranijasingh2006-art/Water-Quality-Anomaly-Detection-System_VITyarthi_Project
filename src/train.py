"""
Training and Evaluation Module for Water Quality Anomaly Detection.
Trains Isolation Forest (primary), Local Outlier Factor (LOF), and One-Class SVM.
Evaluates models, prints statistical summaries, generates graphs in results/,
and saves trained models in models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from preprocess import load_raw_dataset, build_and_apply_preprocessing, save_pipeline

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def train_and_evaluate_models(data_path='data/water_potability.csv', contamination=0.05, random_state=42):
    """
    Main training and evaluation pipeline.
    """
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Load and Preprocess Data
    df_raw = load_raw_dataset(data_path)
    X_scaled, X_raw, y, pipeline_dict, preproc_stats = build_and_apply_preprocessing(df_raw, is_training=True)
    save_pipeline(pipeline_dict, 'models/preprocessing_pipeline.pkl')
    
    print("==================================================")
    print("      DATASET & PREPROCESSING SUMMARY")
    print("==================================================")
    print(f"Original Observations : {preproc_stats['original_shape'][0]}")
    print(f"Original Features     : {preproc_stats['original_shape'][1]}")
    print(f"Total Missing Values  : {preproc_stats['total_missing_values']}")
    print(f"Missing Values Break  : {preproc_stats['missing_per_column']}")
    print(f"Final Features Used   : {len(preproc_stats['feature_names'])} -> {preproc_stats['feature_names']}")
    print(f"Final Observations    : {preproc_stats['final_shape'][0]}")
    print(f"Contamination Rate    : {contamination * 100:.1f}%\n")
    
    # 2. Train Primary Model: Isolation Forest
    print("[1/3] Training Isolation Forest (Primary Model)...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    # scikit-learn convention: 1 for inliers (NORMAL), -1 for outliers (ANOMALOUS)
    if_preds = iso_forest.predict(X_scaled) 
    if_scores = iso_forest.decision_function(X_scaled) # lower score = more anomalous
    
    # Map predictions: 1 -> NORMAL, -1 -> ANOMALOUS
    if_labels = np.where(if_preds == 1, 'NORMAL', 'ANOMALOUS')
    if_anomaly_mask = (if_preds == -1)
    
    joblib.dump(iso_forest, 'models/isolation_forest.pkl')
    print(" -> Saved models/isolation_forest.pkl")
    
    # 3. Train Comparison Model 1: Local Outlier Factor (LOF)
    print("[2/3] Training Local Outlier Factor (LOF)...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )
    lof.fit(X_scaled)
    lof_preds = lof.predict(X_scaled)
    lof_scores = lof.decision_function(X_scaled)
    lof_labels = np.where(lof_preds == 1, 'NORMAL', 'ANOMALOUS')
    lof_anomaly_mask = (lof_preds == -1)
    
    joblib.dump(lof, 'models/lof_model.pkl')
    print(" -> Saved models/lof_model.pkl")
    
    # 4. Train Comparison Model 2: One-Class SVM
    print("[3/3] Training One-Class SVM (OCSVM)...")
    ocsvm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=contamination
    )
    ocsvm.fit(X_scaled)
    ocsvm_preds = ocsvm.predict(X_scaled)
    ocsvm_scores = ocsvm.decision_function(X_scaled)
    ocsvm_labels = np.where(ocsvm_preds == 1, 'NORMAL', 'ANOMALOUS')
    ocsvm_anomaly_mask = (ocsvm_preds == -1)
    
    joblib.dump(ocsvm, 'models/ocsvm_model.pkl')
    print(" -> Saved models/ocsvm_model.pkl\n")
    
    # 5. Statistical Evaluation & Comparison
    n_samples = len(X_scaled)
    if_count = int(np.sum(if_anomaly_mask))
    lof_count = int(np.sum(lof_anomaly_mask))
    ocsvm_count = int(np.sum(ocsvm_anomaly_mask))
    
    overlap_if_lof = int(np.sum(if_anomaly_mask & lof_anomaly_mask))
    overlap_if_ocsvm = int(np.sum(if_anomaly_mask & ocsvm_anomaly_mask))
    overlap_all = int(np.sum(if_anomaly_mask & lof_anomaly_mask & ocsvm_anomaly_mask))
    
    print("==================================================")
    print("      UNSUPERVISED ANOMALY DETECTION RESULTS")
    print("==================================================")
    print(f"Total Water Samples Analyzed : {n_samples}")
    print(f"Isolation Forest Anomalies   : {if_count} ({if_count/n_samples*100:.2f}%)")
    print(f"Local Outlier Factor (LOF)   : {lof_count} ({lof_count/n_samples*100:.2f}%)")
    print(f"One-Class SVM Anomalies      : {ocsvm_count} ({ocsvm_count/n_samples*100:.2f}%)")
    print("--------------------------------------------------")
    print(f"Overlap (IsoForest & LOF)   : {overlap_if_lof} samples ({overlap_if_lof/if_count*100:.1f}% of IF anomalies)")
    print(f"Overlap (IsoForest & OCSVM) : {overlap_if_ocsvm} samples ({overlap_if_ocsvm/if_count*100:.1f}% of IF anomalies)")
    print(f"Overlap (All 3 Models)       : {overlap_all} samples")
    print("--------------------------------------------------")
    print("Isolation Forest Decision Score Summary:")
    print(f"  Mean Score : {if_scores.mean():.4f}")
    print(f"  Std Dev    : {if_scores.std():.4f}")
    print(f"  Min Score  : {if_scores.min():.4f} (Most Anomalous)")
    print(f"  Median     : {np.median(if_scores):.4f}")
    print(f"  Max Score  : {if_scores.max():.4f} (Most Normal)\n")
    
    # 6. Optional Ground-Truth Correlation Analysis
    if y is not None:
        print("==================================================")
        print("   OPTIONAL GROUND-TRUTH POTABILITY CORRELATION")
        print("==================================================")
        print("Note: Unsupervised models were trained WITHOUT labels.")
        print("Evaluating if non-potable water (Potability=0) aligns with detected anomalies:")
        
        # In ground truth, Potability=0 represents non-potable water.
        # Treat non-potable (0) as reference positive anomaly state for reference metrics
        y_non_potable = (y == 0).astype(int)
        if_anom_int = (if_preds == -1).astype(int)
        
        prec = precision_score(y_non_potable, if_anom_int, zero_division=0)
        rec = recall_score(y_non_potable, if_anom_int, zero_division=0)
        f1 = f1_score(y_non_potable, if_anom_int, zero_division=0)
        cm = confusion_matrix(y_non_potable, if_anom_int)
        
        print(f"Reference Alignment Precision : {prec:.4f}")
        print(f"Reference Alignment Recall    : {rec:.4f}")
        print(f"Reference Alignment F1-Score  : {f1:.4f}")
        print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
        print(cm)
        print("Interpretation: Anomaly detection identifies multivariate statistical outliers.")
        print("Statistical anomalies are related to extreme parameter deviations, not solely potability.\n")
        
    # 7. Generate Visualizations
    print("==================================================")
    print("         GENERATING PUBLICATION GRAPHICS")
    print("==================================================")
    
    # Plot 1: Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(if_scores, kde=True, bins=50, color='#2b5c8f', ax=ax, stat='density')
    # Draw threshold line (approx 5th percentile)
    thresh = np.percentile(if_scores, contamination * 100)
    ax.axvline(thresh, color='#d9534f', linestyle='--', linewidth=2, label=f'Contamination Cutoff ({thresh:.3f})')
    ax.set_title('Isolation Forest Decision Score Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Decision Score (Lower = More Anomalous)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('results/anomaly_score_distribution.png', dpi=300)
    plt.close()
    print(" -> Created results/anomaly_score_distribution.png")
    
    # Plot 2: Feature Distribution Comparison (Normal vs Anomalous)
    features_to_plot = preproc_stats['feature_names'][:6] # top 6 features
    df_plot = X_raw[features_to_plot].copy()
    df_plot['Status'] = if_labels
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    colors = {'NORMAL': '#2b5c8f', 'ANOMALOUS': '#d9534f'}
    
    for idx, col in enumerate(features_to_plot):
        sns.boxplot(x='Status', y=col, data=df_plot, ax=axes[idx], palette=colors, hue='Status', legend=False)
        axes[idx].set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(col, fontsize=10)
        
    plt.suptitle('Water Quality Feature Distributions: Normal vs Anomalous', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/feature_distribution.png', dpi=300)
    plt.close()
    print(" -> Created results/feature_distribution.png")
    
    # Plot 3: Model Comparison Bar Chart & Overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar Chart
    model_names = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
    counts = [if_count, lof_count, ocsvm_count]
    bars = ax1.bar(model_names, counts, color=['#2b5c8f', '#41b6c4', '#253494'], width=0.5)
    ax1.set_title('Detected Anomalies by Model (Contamination = 5%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Anomalies Flagged', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height} ({height/n_samples*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    # Overlap Matrix
    overlap_matrix = np.array([
        [if_count, overlap_if_lof, overlap_if_ocsvm],
        [overlap_if_lof, lof_count, int(np.sum(lof_anomaly_mask & ocsvm_anomaly_mask))],
        [overlap_if_ocsvm, int(np.sum(lof_anomaly_mask & ocsvm_anomaly_mask)), ocsvm_count]
    ])
    sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['IsoForest', 'LOF', 'OCSVM'],
                yticklabels=['IsoForest', 'LOF', 'OCSVM'])
    ax2.set_title('Anomaly Inter-Model Overlap Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.close()
    print(" -> Created results/model_comparison.png")
    
    # Plot 4: 2D PCA Visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=if_anomaly_mask,
        cmap=plt.cm.coolwarm,
        alpha=0.6,
        edgecolors='none',
        s=25
    )
    # Draw anomalies highlighted with rings
    anom_pca = X_pca[if_anomaly_mask]
    ax.scatter(anom_pca[:, 0], anom_pca[:, 1], c='red', s=45, label='Flagged Anomaly', edgecolors='black', linewidth=0.5)
    norm_pca = X_pca[~if_anomaly_mask]
    ax.scatter(norm_pca[:200, 0], norm_pca[:200, 1], c='#2b5c8f', s=15, label='Normal Sample', alpha=0.3)
    
    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_title('2D PCA Projection of Water Quality Anomaly Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PCA Component 1 ({var_exp[0]:.1f}% Variance)', fontsize=11)
    ax.set_ylabel(f'PCA Component 2 ({var_exp[1]:.1f}% Variance)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('results/pca_visualization.png', dpi=300)
    plt.close()
    print(" -> Created results/pca_visualization.png\n")
    print("[SUCCESS] Model training and evaluation complete! All artifacts saved.\n")
    
    return {
        'n_samples': n_samples,
        'if_count': if_count,
        'lof_count': lof_count,
        'ocsvm_count': ocsvm_count,
        'if_scores': if_scores,
        'feature_names': preproc_stats['feature_names']
    }

if __name__ == '__main__':
    train_and_evaluate_models()
