"""
FKC Credit Default Prediction System
Main Streamlit Application - Home Page
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FKC Credit Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💳 FKC Credit Default Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Frontier Kenya Credit Limited - AI-Powered Risk Assessment</div>', unsafe_allow_html=True)

# Student Branding
st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 15px; border-radius: 10px; margin: 20px auto; max-width: 600px; color: white;">
    <p style="margin: 5px 0; font-size: 1.1rem; font-weight: bold;">By Michael Kamau Kibugu</p>
    <p style="margin: 5px 0; font-size: 1rem;">Student ID: 093371</p>
    <p style="margin: 5px 0; font-size: 1rem;">Strathmore University</p>
    <p style="margin: 5px 0; font-size: 1rem;">Nairobi, Kenya</p>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    ### 🎯 Welcome to the FKC Credit Risk Assessment Platform
    
    This intelligent system uses **Machine Learning** to predict credit default risk for 
    Frontier Kenya Credit Limited customers. Built with advanced analytics and trained on 
    realistic customer data, it helps make informed lending decisions.
    """)

st.markdown("---")

# Key Features
st.markdown("### ✨ Key Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                padding: 20px; border-radius: 10px; color: white; min-height: 180px;">
        <h3 style="color: white; margin-top: 0;">📊 Data Generation</h3>
        <p style="color: white; line-height: 1.6;">Generate realistic customer credit data with 2,500 records including demographics, 
        financial history, and loan details.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); 
                padding: 20px; border-radius: 10px; color: white; min-height: 180px;">
        <h3 style="color: white; margin-top: 0;">🔍 EDA Analysis</h3>
        <p style="color: white; line-height: 1.6;">Comprehensive exploratory data analysis with interactive visualizations and 
        statistical insights.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%); 
                padding: 20px; border-radius: 10px; color: white; min-height: 180px;">
        <h3 style="color: white; margin-top: 0;">🤖 ML Models</h3>
        <p style="color: white; line-height: 1.6;">Train and compare 3 machine learning models: Logistic Regression, Random Forest, 
        and SVM.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; min-height: 180px;">
        <h3 style="color: white; margin-top: 0;">🎯 Predictions</h3>
        <p style="color: white; line-height: 1.6;">Make real-time credit default predictions with probability scores and risk 
        assessment.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# System Overview
st.markdown("### 📋 System Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🏦 About FKC
    
    **Frontier Kenya Credit Limited** operates across 5 key locations in Kenya:
    - 🏙️ Nairobi (Main Branch)
    - 🏘️ Kayole
    - 🏘️ Kamulu
    - 🛣️ Mombasa Road
    - 🌾 Mwea
    
    #### 💰 Loan Products
    
    - **School Fees Loan** (14.5% interest)
    - **Utility Loan** (16.0% interest)
    - **Cash Loan** (18.5% interest)
    - **Business Capital Loan** (19.5% interest)
    - **Emergency Loan** (22.0% interest)
    """)

with col2:
    st.markdown("""
    #### 🎓 Customer Profiles
    
    The system analyzes 5 distinct customer segments:
    
    1. **Stable Professional** (15% default rate)
       - Formally employed, higher income
       
    2. **Young Entrepreneur** (25% default rate)
       - Self-employed, moderate income
       
    3. **Hustler/Casual Worker** (45% default rate)
       - Casual labor, lower income
       
    4. **Overburdened Family Person** (40% default rate)
       - Multiple financial obligations
       
    5. **Entry Level Worker** (20% default rate)
       - Young professionals, growing income
    """)

st.markdown("---")

# Quick Stats
st.markdown("### 📊 System Statistics")

# Check if data exists
data_path = Path("data/FKC_Credit_System_Data.csv")
model_path = Path("models/best_model.pkl")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.metric("Total Records", f"{len(df):,}")
    else:
        st.metric("Total Records", "No Data")

with col2:
    if data_path.exists():
        defaulters = len(df[df['Is_Defaulter'] == 'Yes'])
        st.metric("Defaulters", f"{defaulters:,}", f"{defaulters/len(df)*100:.1f}%")
    else:
        st.metric("Defaulters", "N/A")

with col3:
    st.metric("ML Models", "3", "LR, RF, SVM")

with col4:
    if model_path.exists():
        st.metric("Model Status", "✅ Trained", "Ready")
    else:
        st.metric("Model Status", "⚠️ Not Trained", "Train First")

st.markdown("---")

# Getting Started
st.markdown("### 🚀 Getting Started")

st.markdown("""
Follow these steps to use the system:

1. **📊 Generate Data** - Go to the Data Generation page to create customer records
2. **🔍 Explore Data** - Analyze the data with interactive visualizations in EDA Analysis
3. **🤖 Train Models** - Train and compare machine learning models
4. **🎯 Make Predictions** - Use the trained model to predict default risk for new customers

Use the **sidebar navigation** to access different pages of the application.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>FKC Credit Default Prediction System</strong></p>
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
    <p style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
        <strong>Developed by:</strong> Michael Kamau Kibugu (093371)<br>
        Strathmore University, Nairobi, Kenya
    </p>
    <p style="margin-top: 10px; font-size: 0.9rem;">© 2026 Frontier Kenya Credit Limited</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Navigation Guide")
    st.markdown("""
    **Use the pages above to:**
    
    - 📊 Generate synthetic data
    - 🔍 Perform EDA analysis
    - 🤖 Train ML models
    - 🎯 Make predictions
    
    ---
    
    ### ℹ️ System Info
    """)
    
    if data_path.exists():
        st.success("✅ Data Available")
    else:
        st.warning("⚠️ No Data - Generate First")
    
    if model_path.exists():
        st.success("✅ Model Trained")
    else:
        st.warning("⚠️ Model Not Trained")
    
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("For questions or issues, contact the development team.")
