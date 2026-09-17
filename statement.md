# Project Statement & Problem Scope

## 1. Problem Statement
Water quality monitoring is crucial for public health, environmental safety, and industrial processes. Traditional threshold-based monitoring methods (e.g., checking if $\text{pH} > 8.5$) evaluate each water parameter independently. However, water safety depends on complex, **multivariate interactions** across chemical and physical properties (e.g., pH, Turbidity, Total Dissolved Solids, Sulfate, Chloramines, Conductivity). Single-parameter rules fail to detect subtle anomalies where individual parameters remain within broad acceptable bounds, but their combination indicates significant contamination or equipment malfunction.

This project addresses the challenge by implementing an **AI-Based Water Quality Anomaly Detection System using Machine Learning**. The system applies unsupervised learning algorithms—specifically **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM**—to learn the baseline structural geometry of normal multivariate water measurements and automatically flag anomalous observations without requiring pre-labeled training data.

---

## 2. Scope of the Project
- **In-Scope**:
  - Ingestion and cleaning of 9 numerical water quality parameters (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`).
  - Automated non-leakage preprocessing pipeline (median missing-value imputation and feature standardization).
  - Unsupervised model training on multivariate feature distributions using Isolation Forest as the primary model.
  - Comparative evaluation against Local Outlier Factor (density-based) and One-Class SVM (boundary-based).
  - Computation of continuous anomaly/decision scores and anomaly predictions (`NORMAL` vs `ANOMALOUS`).
  - Generation of publication-quality statistical charts and 2D PCA projection visual maps.
  - Interactive Command-Line Interface (`main.py`) with input validation and authentic dataset example testing.
  - Interactive Web Dashboard (`app.py`) for visual parameter manipulation and real-time inference.
  - Unit and integration validation test suite (`tests/`).

- **Out-of-Scope**:
  - Direct biological toxicity assaying or chemical laboratory testing.
  - Real-time hardware IoT sensor network hardware integration (simulated via tabular measurements).

---

## 3. Target Users
1. **Environmental Quality Inspectors & Analysts**: To rapidly scan multivariate water sample data from rivers, reservoirs, or treatment plants for unusual parameter combinations.
2. **Water Treatment Plant Operators**: To receive automated alerts regarding potential chemical dosing anomalies or pipe degradation.
3. **Academic Researchers & Students**: To study and demonstrate unsupervised machine learning methodologies, model comparison techniques, and non-leakage preprocessing pipelines.

---

## 4. High-Level Features
- **Multivariate Anomaly Detection**: Evaluates 9 physical/chemical parameters simultaneously using Isolation Forest.
- **Multi-Model Benchmark**: Cross-verifies detected anomalies across Local Outlier Factor and One-Class SVM.
- **Robust Preprocessing Pipeline**: Handles missing values dynamically using median imputation and standardizes feature ranges cleanly without data leakage.
- **Parameter Deviation Analysis**: Identifies specific parameters in an observation that diverge significantly from normal baseline percentiles.
- **Dual User Interface**: Provides both a Command-Line Interface (`main.py`) and a Streamlit Web Dashboard (`app.py`).
- **Comprehensive Visualization**: Generates 4 publication graphics including decision score KDE plots, feature boxplots, overlap matrices, and 2D PCA scatter projections.
