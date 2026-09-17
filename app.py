"""
Streamlit Web Dashboard for Water Quality Anomaly Detection.
Provides interactive visual controls, real-time prediction,
and model score visualization.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit is not installed. Install with: pip install streamlit")
    sys.exit(1)

from predict import WaterQualityPredictor
from utils import get_example_observations, FEATURE_METADATA

st.set_page_config(
    page_title="Water Quality Anomaly Detection",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_predictor():
    return WaterQualityPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}. Please run `python src/train.py` first.")
    st.stop()

st.title("💧 AI-Based Water Quality Anomaly Detection System")
st.markdown("""
This system uses **Unsupervised Machine Learning (Isolation Forest)** to evaluate multivariate water quality parameters 
and determine whether an observation represents **NORMAL** water measurements or an **ANOMALOUS** statistical outlier.
""")

st.sidebar.header("⚙️ Input Mode")
mode = st.sidebar.radio("Choose Input Method", ["Manual Parameter Controls", "Preset Dataset Examples"])

input_data = {}

if mode == "Manual Parameter Controls":
    st.sidebar.subheader("Adjust Water Parameters")
    stats = predictor.feature_stats
    
    for feat in predictor.feature_names:
        meta = FEATURE_METADATA.get(feat, {'unit': '', 'desc': feat})
        default_val = float(stats[feat]['median'])
        min_val = float(stats[feat]['min'])
        max_val = float(stats[feat]['max'])
        
        # Expand slider range slightly beyond dataset min/max for extreme anomaly testing
        slider_min = max(0.0, min_val * 0.5)
        slider_max = max_val * 1.5
        
        input_data[feat] = st.sidebar.slider(
            f"{feat} ({meta['unit']})",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=default_val,
            help=meta['desc']
        )
else:
    st.sidebar.subheader("Select Dataset Example")
    example_choice = st.sidebar.selectbox("Choose Example", ["Normal Observation", "Anomalous Observation"])
    normal_obs, anom_obs = get_example_observations()
    input_data = normal_obs if example_choice == "Normal Observation" else anom_obs

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Water Quality Measurements")
    df_input = pd.DataFrame([input_data]).T.reset_index()
    df_input.columns = ["Parameter", "Value"]
    df_input["Unit"] = df_input["Parameter"].apply(lambda x: FEATURE_METADATA.get(x, {}).get('unit', ''))
    st.dataframe(df_input, use_container_width=True)

with col2:
    st.subheader("📊 Model Anomaly Diagnosis")
    if st.button("🔍 Analyze Observation", type="primary", use_container_width=True):
        res = predictor.predict_observation(input_data)
        
        if res['prediction'] == 'NORMAL':
            st.success("### Status: NORMAL WATER SAMPLE")
        else:
            st.error("### Status: ANOMALOUS OBSERVATION DETECTED")
            
        st.metric("Isolation Forest Decision Score", f"{res['anomaly_score']:+.4f}", help="Negative scores indicate high degree of anomaly")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Local Outlier Factor**: {res['lof_prediction']}")
        with col_b:
            st.info(f"**One-Class SVM**: {res['ocsvm_prediction']}")
            
        st.markdown("#### Parameter Deviation Analysis:")
        if res['parameter_deviations']:
            for dev in res['parameter_deviations']:
                st.warning(f"⚠️ {dev}")
        else:
            st.success("✅ All parameter values align with multivariate normal baseline distributions.")

st.markdown("---")
st.subheader("📈 Pre-computed Model Visualizations")
tab1, tab2, tab3, tab4 = st.tabs(["Anomaly Score Distribution", "Feature Distributions", "Model Comparison", "2D PCA Projection"])

with tab1:
    if os.path.exists("results/anomaly_score_distribution.png"):
        st.image("results/anomaly_score_distribution.png", caption="Isolation Forest Decision Score Distribution")
with tab2:
    if os.path.exists("results/feature_distribution.png"):
        st.image("results/feature_distribution.png", caption="Parameter Distributions (Normal vs Anomalous)")
with tab3:
    if os.path.exists("results/model_comparison.png"):
        st.image("results/model_comparison.png", caption="Model Comparison & Inter-Model Overlap Matrix")
with tab4:
    if os.path.exists("results/pca_visualization.png"):
        st.image("results/pca_visualization.png", caption="2D PCA Scatter Plot of Water Observations")
