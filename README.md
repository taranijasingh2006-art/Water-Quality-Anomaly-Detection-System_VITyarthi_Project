# AI-Based Water Quality Anomaly Detection System using Machine Learning

An academic, reproducible, unsupervised machine learning project designed to detect anomalous or unusual water-quality observations from multivariate sensor measurements.

---

## 1. Project Title
**AI-Based Water Quality Anomaly Detection System using Machine Learning**

## 2. Project Overview
Water quality assessment is critical for environmental monitoring, municipal public health, and industrial process control. Traditional water quality evaluation often relies on single-parameter threshold rules (e.g., checking if $\text{pH} > 8.5$). However, water safety and chemical stability depend on complex **multivariate interactions** across physical and chemical properties.

This project implements an **unsupervised machine-learning system** that learns normal multivariate patterns from real-world water-quality data. It detects observations that deviate significantly from learned baseline distributions using **Isolation Forest** as the primary model, alongside comparative evaluations using **Local Outlier Factor (LOF)** and **One-Class Support Vector Machine (One-Class SVM)**.

## 3. Problem Statement
Detecting water contamination or sensor degradation manually across multi-parameter monitoring stations is inefficient and prone to missed events. Anomalies in water quality can arise from chemical spills, pipe corrosion, industrial discharge, or equipment failure. A machine learning model must identify these unusual multi-parameter observations automatically without relying on expensive or hard-to-obtain labeled anomaly data.

## 4. Motivation
- **Multivariate Dynamics**: Water quality cannot be judged purely by individual parameter thresholds; combinations of parameters (e.g., normal pH combined with abnormally high conductivity and turbidity) indicate anomalies.
- **Label Scarcity**: Real-world anomaly datasets rarely have exhaustive ground-truth anomaly labels. Unsupervised anomaly detection algorithms learn the underlying structure directly from unlabelled observation space.
- **Academic Rigor**: Demonstrates correct ML practices—preventing data leakage, applying statistical preprocessing, comparing multiple model paradigms, generating actual data-driven visualizations, and delivering an interactive CLI application.

## 5. Objectives
1. Build a non-leakage preprocessing pipeline for water quality feature matrices.
2. Train an unsupervised **Isolation Forest** model to compute anomaly predictions and continuous decision scores.
3. Compare performance against **Local Outlier Factor (LOF)** and **One-Class SVM**.
4. Generate 4 publication-quality visualization figures (`results/*.png`).
5. Provide a interactive Command-Line Interface (`main.py`) with input validation and authentic dataset example tests.

## 6. Dataset
- **Dataset**: Kaggle Water Potability Dataset (`water_potability.csv`)
- **Total Observations**: 3,276 water samples
- **Original Features**: 10 columns (9 numerical water-quality parameters + 1 optional binary label `Potability`)

## 7. Dataset Source
- **Public Repository**: Hosted on Kaggle / Open Data Mirrors (`https://raw.githubusercontent.com/fenago/datasets/main/water_potability.csv`)
- **Citation**: Kadiwal, A. (Kaggle Water Quality Dataset).

## 8. Features Used
The system evaluates 9 scientifically meaningful numerical features:

| Feature | Description | Units | Dataset Range | Missing Values |
|---|---|---|---|---|
| `ph` | Acidity / Alkalinity level | pH units | 0.00 - 14.00 | 491 |
| `Hardness` | Calcium and magnesium concentration | mg/L | 47.43 - 323.12 | 0 |
| `Solids` | Total Dissolved Solids (TDS) | ppm | 320.94 - 61227.20 | 0 |
| `Chloramines` | Disinfectant compound concentration | ppm | 0.35 - 13.13 | 0 |
| `Sulfate` | Dissolved sulfate salts | mg/L | 129.00 - 481.03 | 781 |
| `Conductivity` | Electrical conductivity | μS/cm | 201.44 - 753.34 | 0 |
| `Organic_carbon` | Total organic carbon level | ppm | 2.20 - 28.30 | 0 |
| `Trihalomethanes` | Chlorination byproduct concentration | μg/L | 0.74 - 124.00 | 162 |
| `Turbidity` | Measure of water cloudiness | NTU | 1.45 - 6.74 | 0 |

> **Note**: Ground-truth column `Potability` (1,278 potable / 1,998 non-potable) is **excluded from model training** and used strictly for optional reference correlation analysis.

## 9. Technologies Used
- **Language**: Python 3.13+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Model Serialization**: `joblib`
- **Data Visualization**: `matplotlib`, `seaborn`

## 10. Project Structure
```
AI-Water-Quality-Anomaly-Detection/
│
├── data/
│   └── water_potability.csv           # Authentic public water dataset
│
├── models/
│   ├── isolation_forest.pkl          # Saved primary trained model
│   ├── lof_model.pkl                 # Saved LOF comparison model
│   ├── ocsvm_model.pkl               # Saved One-Class SVM comparison model
│   └── preprocessing_pipeline.pkl    # Imputer + Scaler + Feature metadata
│
├── results/
│   ├── anomaly_score_distribution.png # Histogram/KDE of decision scores
│   ├── feature_distribution.png      # Feature comparison boxplots
│   ├── model_comparison.png          # Model anomaly counts & overlap matrix
│   └── pca_visualization.png         # 2D PCA projection scatter plot
│
├── src/
│   ├── preprocess.py                 # Non-leakage preprocessing & pipeline
│   ├── train.py                      # Unsupervised model training & evaluation
│   ├── predict.py                    # Inference engine class
│   └── utils.py                      # Authentic sample retrieval & helpers
│
├── main.py                           # Interactive CLI Application
├── requirements.txt                  # Exact project dependencies
├── README.md                         # Detailed project documentation
└── .gitignore                        # Git exclusion rules
```

## 11. Machine Learning Methodology
The project strictly follows **Unsupervised Anomaly Detection**:
```
Raw Water Samples (3,276 rows x 9 features)
              │
              ▼
   Median Imputation (Handling Missing Values)
              │
              ▼
   StandardScaler (Zero Mean, Unit Variance)
              │
              ├──► Primary Model: Isolation Forest
              ├──► Comparison Model 1: Local Outlier Factor (LOF)
              └──► Comparison Model 2: One-Class SVM
              │
              ▼
    Anomaly Predictions (-1: ANOMALOUS, 1: NORMAL)
    & Continuous Decision / Anomaly Scores
```

## 12. Why Anomaly Detection?
Unlike supervised classification which requires pre-labeled categories, anomaly detection identifies observations that stand out from the normal structural geometry of feature space. In environmental monitoring:
1. Outliers indicate unexpected physical/chemical behavior.
2. Anomaly detection catches novel failure modes never seen before in training data.
3. The system evaluates all parameters simultaneously in 9-dimensional space.

## 13. Data Preprocessing
Preprocessing is executed deterministically without data leakage:
- **Missing Value Imputation**: Uses `SimpleImputer(strategy='median')`. Median imputation is robust against extreme values.
- **Feature Scaling**: Uses `StandardScaler()` to standardize feature ranges for distance-based models (LOF/OCSVM).
- **Pipeline Preservation**: The fitted imputer and scaler are saved together into `models/preprocessing_pipeline.pkl` so that test inputs undergo identical transformations.

## 14. Isolation Forest
**Isolation Forest** is an ensemble tree-based algorithm designed specifically for anomaly detection:
- **Principle**: Instead of profiling normal points, it explicitly isolates anomalies. Because anomalies are sparse and different, they require fewer random partitioning splits to isolate in a decision tree.
- **Path Length**: Anomalous points have significantly shorter path lengths from the root node to terminal leaves.
- **Suitability**: Scale-invariant, computationally efficient ($O(n \log n)$), and highly effective in high-dimensional feature spaces.

## 15. Local Outlier Factor (LOF)
**Local Outlier Factor (LOF)** is a density-based unsupervised anomaly detection algorithm:
- **Principle**: Measures the local density deviation of a given data point relative to its $k$-nearest neighbors.
- **Suitability**: Points that have a substantially lower local density than their neighbors are flagged as local outliers.

## 16. Optional One-Class SVM
**One-Class SVM** constructs a tight decision boundary around the high-density region of normal training points in a kernelized feature space (RBF kernel), flagging points outside the boundary as anomalies.

## 17. Evaluation Methodology
Because training is unsupervised, evaluation uses:
1. **Anomaly Flag Count & Percentage**: Quantifies detected outliers at fixed contamination rate ($\alpha = 0.05$).
2. **Decision Score Statistics**: Evaluates mean, standard deviation, min (most anomalous), median, and max (most normal) scores.
3. **Inter-Model Overlap**: Computes agreement percentage across Isolation Forest, LOF, and One-Class SVM.
4. **Visual Inspection**: PCA 2D projections and feature boxplots.
5. **Reference Ground-Truth Correlation**: Evaluates alignment against non-potable samples without label leak during training.

## 18. Actual Results (Generated from Code Execution)

> [!IMPORTANT]
> All figures below represent actual calculated results from running `python src/train.py`.

- **Total Samples Analyzed**: 3,276
- **Contamination Rate Parameter**: 5.0%

| Model | Anomalies Flagged | Percentage Flagged | Min Score (Most Anomalous) | Max Score (Most Normal) |
|---|---|---|---|---|
| **Isolation Forest** (Primary) | **164** | **5.01%** | **-0.1033** | **+0.1441** |
| **Local Outlier Factor** (LOF) | 132 | 4.03% | -0.6387 | +0.1872 |
| **One-Class SVM** (OCSVM) | 170 | 5.19% | -0.1983 | +9.2415 |

### Inter-Model Overlap Analysis
- **IsoForest & LOF Agreement**: 90 samples (**54.9%** of IF anomalies)
- **IsoForest & One-Class SVM Agreement**: 111 samples (**67.7%** of IF anomalies)
- **Consensus Anomalies (All 3 Models)**: **78 samples**

### Generated Graphics Artifacts
1. `results/anomaly_score_distribution.png`: Shows Isolation Forest decision score distribution centered at +0.0748 with a cutoff threshold at -0.003.
2. `results/feature_distribution.png`: Illustrates parameter boxplots where anomalous observations display extreme values (e.g. TDS $> 50,000$ ppm or pH $< 4.0$ / $> 10.0$).
3. `results/model_comparison.png`: Bar chart comparison and heatmapped overlap matrix.
4. `results/pca_visualization.png`: 2D PCA scatter plot showing clear separation of isolated anomaly points.

## 19. Installation

### Requirements
- Python 3.10+ (Tested on Python 3.13.7)
- Operating System: Windows / Linux / macOS

## 20. Virtual Environment Setup

### Windows (PowerShell / Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 21. Dependency Installation
```bash
pip install -r requirements.txt
```

## 22. Training Instructions
To train the Isolation Forest, LOF, and One-Class SVM models, compute metrics, and generate graphics:
```bash
python src/train.py
```

## 23. Prediction Instructions
To run inference via Python code:
```python
from src.predict import WaterQualityPredictor

predictor = WaterQualityPredictor()
sample = {
    'ph': 3.5,            # Abnormally low pH
    'Hardness': 195.0,
    'Solids': 55000.0,    # Abnormally high TDS
    'Chloramines': 7.1,
    'Sulfate': 330.0,
    'Conductivity': 420.0,
    'Organic_carbon': 14.0,
    'Trihalomethanes': 65.0,
    'Turbidity': 6.5
}
result = predictor.predict_observation(sample)
print(result['prediction'])      # Output: ANOMALOUS
print(result['anomaly_score'])   # Output: negative score e.g. -0.0852
```

## 24. Command-Line Interface (CLI) Examples
Run the terminal application:
```bash
python main.py
```

### CLI Menu Preview
```
==================================================
   AI WATER QUALITY ANOMALY DETECTION SYSTEM
==================================================
CHOOSE AN OPTION:
 1. Enter water-quality values manually
 2. Use an example NORMAL water sample (from dataset)
 3. Use an example ANOMALOUS water sample (from dataset)
 4. View model summary & dataset statistics
 5. Exit
```

## 25. Scientific Limitations
- **Educational System**: Designed for academic ML demonstration.
- **Non-Diagnostic**: An anomaly flag indicates statistical divergence from dataset patterns; it does **not** automatically prove that water is toxic or unsafe for human consumption.
- **Sensor Drift vs Real Contamination**: Anomalies can be caused by physical sensor failure as well as actual water pollution.

## 26. Future Scope
1. Integrate time-series sequence models (LSTM / Autoencoders) for temporal water monitoring.
2. Incorporate streaming pipeline tools (Apache Kafka) for real-time sensor ingestion.
3. Add explainable AI (SHAP / LIME) to highlight feature contribution weights per anomaly.

## 27. Conclusion
This project successfully demonstrates an end-to-end, reproducible, unsupervised water-quality anomaly detection system using Isolation Forest. The model effectively identifies multivariate statistical outliers, supported by LOF and One-Class SVM comparisons, non-leakage preprocessing, data visualizations, and an interactive CLI application.

---

## Academic Viva Preparation Q&A

### 1. What is anomaly detection?
Anomaly detection is the identification of rare items, events, or observations that raise suspicions by differing significantly from the majority of the data.

### 2. Why is this an unsupervised learning problem?
Because in real-world water quality monitoring, explicit labels for every possible type of contamination or anomaly are unavailable. Unsupervised models learn the natural distribution of normal data and flag points that do not conform.

### 3. What is Isolation Forest?
Isolation Forest is an unsupervised tree-based ensemble algorithm that isolates anomalies by randomly selecting a feature and randomly selecting a split value between the feature's minimum and maximum values.

### 4. How does Isolation Forest identify anomalies?
Random partitioning produces noticeably shorter paths for anomalies because fewer conditions are required to isolate unusual points that differ from the dense cluster of normal points.

### 5. What is contamination?
The contamination parameter defines the expected proportion of anomalies in the dataset (e.g., 0.05 = 5%). It sets the decision score threshold for partitioning points into NORMAL vs ANOMALOUS.

### 6. Why is preprocessing important?
Preprocessing cleans missing values and standardizes feature scaling, preventing invalid numerical entries or missing values from crashing the model or skewing model distances.

### 7. Why might feature scaling matter?
Distance-based and boundary-based models (LOF and One-Class SVM) rely on Euclidean distances. Unscaled features with large numerical magnitudes (e.g., TDS in tens of thousands) would dominate features with small magnitudes (e.g., pH from 0 to 14).

### 8. What is Local Outlier Factor (LOF)?
LOF is a density-based anomaly detection algorithm that compares the local density of an observation with the local densities of its $k$-nearest neighbors.

### 9. What is the difference between classification and anomaly detection?
Classification maps inputs to predefined target classes using labeled training data. Anomaly detection models the normal structure of data without relying on target labels and identifies outliers that deviate from that structure.

### 10. What is an anomaly score?
An anomaly score is a continuous numeric output produced by the model (e.g., Isolation Forest decision function) quantifying how normal or anomalous an observation is.

### 11. Why is anomaly score NOT necessarily a probability?
Anomaly scores are decision function outputs (or path length metrics) derived from tree depths or density ratios. They are not calibrated probabilities constrained to $[0, 1]$ and do not represent probability distribution functions.

### 12. How are anomalies evaluated when ground-truth labels do not exist?
Through unsupervised metrics: anomaly count, contamination percentage, score distributions, inter-model agreement overlap, and visual inspections (PCA 2D projection and feature boxplots).

### 13. What are the limitations of the model?
The model relies on static tabular data without temporal sequence modeling and assumes the training set represents typical baseline water quality.

### 14. Why does an anomalous observation not automatically mean unsafe water?
An anomaly signifies statistical divergence from historical observations (e.g. unusually high mineral content or unusual pH/conductivity combo). High mineral content can be safe, or a sensor may be miscalibrated. Physical chemical analysis is required to verify safety.
