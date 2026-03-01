"""
Credit Card Fraud Detection - Streamlit App
===========================================
A web application for predicting fraudulent credit card transactions.

Author: Data Scientist
"""

import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import joblib

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================================
# MODEL LOADING / TRAINING
# ============================================================================
@st.cache_resource
def train_model():
    """
    Train a simple XGBoost model for fraud detection.
    In production, load from saved model file.
    """
    # Generate synthetic training data (similar to main.py)
    np.random.seed(42)
    n_samples = 50000
    fraud_ratio = 0.0017
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Generate legitimate transactions
    legitimate_data = {
        'Time': np.random.uniform(0, 172800, n_legitimate),
        'Amount': np.random.exponential(80, n_legitimate),
    }
    for i in range(1, 29):
        legitimate_data[f'V{i}'] = np.random.normal(0, 1, n_legitimate)
    
    # Generate fraudulent transactions
    fraud_data = {
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Amount': np.random.exponential(150, n_fraud),
    }
    for i in range(1, 29):
        fraud_data[f'V{i}'] = np.random.normal(0.5, 1.2, n_fraud)
    
    # Create DataFrames
    df_legitimate = pd.DataFrame(legitimate_data)
    df_legitimate['Class'] = 0
    
    df_fraud = pd.DataFrame(fraud_data)
    df_fraud['Class'] = 1
    
    # Combine
    df = pd.concat([df_legitimate, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)
    
    return model

# Train the model
model = train_model()

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.title("💳 Credit Card Fraud Detection")
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts whether a credit card transaction is fraudulent 
    based on various transaction features.
    
    **Risk Levels:**
    - 🟢 Low: Fraud probability < 20%
    - 🟡 Medium: Fraud probability 20-50%
    - 🔴 High: Fraud probability > 50%
    """
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Transaction Details")
    st.markdown("Enter the transaction features below:")
    
    # Create input fields for features
    with st.container():
        # Time and Amount
        col_time, col_amount = st.columns(2)
        with col_time:
            time_val = st.number_input("Time (seconds from first transaction)", 
                                      min_value=0.0, max_value=172800.0, 
                                      value=50000.0, step=100.0)
        with col_amount:
            amount_val = st.number_input("Transaction Amount ($)", 
                                         min_value=0.0, max_value=2500.0, 
                                         value=50.0, step=1.0)
        
        st.markdown("### V1-V28 Features (PCA Components)")
        
        # Create rows for V1-V28
        v_features = {}
        for i in range(1, 29):
            col = st.columns(4)[(i-1) % 4]
            with col:
                v_features[f'V{i}'] = st.number_input(
                    f"V{i}", 
                    value=0.0, 
                    step=0.1,
                    format="%.2f"
                )

with col2:
    st.header("Prediction")
    st.markdown("###")
    
    # Predict button
    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        # Prepare input features
        features = {
            'Time': time_val,
            'Amount': amount_val,
        }
        features.update(v_features)
        
        # Create DataFrame for prediction
        X_pred = pd.DataFrame([features])
        
        # Get prediction and probability
        prediction = model.predict(X_pred)[0]
        fraud_probability = model.predict_proba(X_pred)[0][1]
        
        # Display results
        st.markdown("---")
        st.subheader("Results")
        
        # Probability gauge
        st.metric("Fraud Probability", f"{fraud_probability*100:.2f}%")
        
        # Progress bar for probability
        progress_bar = st.progress(fraud_probability)
        
        # Risk level indicator
        if fraud_probability < 0.2:
            risk_level = "🟢 LOW"
            risk_color = "green"
        elif fraud_probability < 0.5:
            risk_level = "🟡 MEDIUM"
            risk_color = "orange"
        else:
            risk_level = "🔴 HIGH"
            risk_color = "red"
        
        st.markdown(f"### Risk Level: **{risk_level}**")
        
        # Prediction result
        if prediction == 1:
            st.error("⚠️ This transaction is likely **FRAUDULENT**!")
        else:
            st.success("✅ This transaction appears to be **LEGITIMATE**.")
        
        # Explanation
        with st.expander("📋 View Details"):
            st.write(f"**Model Prediction:** {'Fraud' if prediction == 1 else 'Legitimate'}")
            st.write(f"**Fraud Probability:** {fraud_probability*100:.4f}%")
            st.write(f"**Legitimate Probability:** {(1-fraud_probability)*100:.4f}%")

# Feature importance section
st.markdown("---")
st.header("Feature Importance")
st.info("These are the most important features used by the model to detect fraud.")

# Get feature importance from model
feature_importance = pd.DataFrame({
    'feature': model.feature_names_in_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Display top 10 features
top_features = feature_importance.head(10)
st.bar_chart(top_features.set_index('feature')['importance'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Credit Card Fraud Detection Model | Built with XGBoost & Streamlit</p>
</div>
""", unsafe_allow_html=True)
