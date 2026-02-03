"""
FKC Credit Default Prediction System
Page 2: EDA Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="EDA Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Exploratory Data Analysis")
st.markdown("Comprehensive analysis of FKC customer credit data")

st.markdown("---")

# Load data
data_path = Path("data/FKC_Credit_System_Data.csv")

if not data_path.exists():
    st.warning("⚠️ No data file found. Please generate data first from the Data Generation page.")
    st.stop()

# Load the data
df = pd.read_csv(data_path)

# Create binary target
df['Default_Binary'] = (df['Is_Defaulter'] == 'Yes').astype(int)

# Feature engineering
df['Debt_to_Income_Ratio'] = df['Outstanding_Balance'] / (df['Estimated_Monthly_Income'] + 1)
df['Payment_Rate'] = df['Payments_Made'] / (df['Payments_Made'] + df['Payments_Missed'] + 0.01)
df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Estimated_Monthly_Income'] + 1)

st.success(f"✅ Loaded {len(df):,} customer records with {len(df.columns)} features")

# Quick Stats
st.markdown("### 📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    defaulters = len(df[df['Is_Defaulter'] == 'Yes'])
    st.metric("Defaulters", f"{defaulters:,}", f"{defaulters/len(df)*100:.1f}%")

with col3:
    non_defaulters = len(df[df['Is_Defaulter'] == 'No'])
    st.metric("Non-Defaulters", f"{non_defaulters:,}", f"{non_defaulters/len(df)*100:.1f}%")

with col4:
    avg_loan = df['Loan_Amount'].mean()
    st.metric("Avg Loan", f"KES {avg_loan:,.0f}")

with col5:
    total_outstanding = df['Outstanding_Balance'].sum()
    st.metric("Total Outstanding", f"KES {total_outstanding/1e6:.1f}M")

st.markdown("---")

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Target Analysis", 
    "📊 Feature Distributions", 
    "🔗 Correlations",
    "🎯 Default Analysis",
    "📋 Data Quality"
])

# Tab 1: Target Analysis
with tab1:
    st.markdown("### 🎯 Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Default distribution pie chart
        fig = px.pie(
            df, 
            names='Is_Defaulter', 
            title='Default Status Distribution',
            color='Is_Defaulter',
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc66'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Default count bar chart
        default_counts = df['Is_Defaulter'].value_counts()
        fig = px.bar(
            x=default_counts.index,
            y=default_counts.values,
            labels={'x': 'Default Status', 'y': 'Count'},
            title='Default Status Count',
            color=default_counts.index,
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Class balance info
    st.info(f"""
    **Class Balance Analysis:**
    - Non-Defaulters: {non_defaulters:,} ({non_defaulters/len(df)*100:.1f}%)
    - Defaulters: {defaulters:,} ({defaulters/len(df)*100:.1f}%)
    - Balance Ratio: {non_defaulters/defaulters:.2f}:1
    - **Conclusion:** Slight class imbalance but manageable for modeling
    """)

# Tab 2: Feature Distributions
with tab2:
    st.markdown("### 📊 Feature Distributions")
    
    # Numerical features
    st.markdown("#### 🔢 Numerical Features")
    
    numerical_cols = ['Age', 'Estimated_Monthly_Income', 'Loan_Amount', 
                     'Outstanding_Balance', 'Overdue_Amount', 'Payments_Made', 
                     'Payments_Missed', 'Number_of_Loans']
    
    selected_num_feature = st.selectbox("Select numerical feature to analyze:", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df, 
            x=selected_num_feature, 
            color='Is_Defaulter',
            title=f'{selected_num_feature} Distribution by Default Status',
            marginal='box',
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc66'},
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, 
            x='Is_Defaulter', 
            y=selected_num_feature,
            title=f'{selected_num_feature} by Default Status',
            color='Is_Defaulter',
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("**Statistical Summary:**")
    summary_stats = df.groupby('Is_Defaulter')[selected_num_feature].describe()
    st.dataframe(summary_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Categorical features
    st.markdown("#### 📝 Categorical Features")
    
    categorical_cols = ['Location', 'Employment_Status', 'Monthly_Income_Bracket', 
                       'Loan_Product', 'Term_Type', 'Loan_Status', 'Gender']
    
    selected_cat_feature = st.selectbox("Select categorical feature to analyze:", categorical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count plot
        cat_counts = df[selected_cat_feature].value_counts()
        fig = px.bar(
            x=cat_counts.index,
            y=cat_counts.values,
            title=f'{selected_cat_feature} Distribution',
            labels={'x': selected_cat_feature, 'y': 'Count'},
            color=cat_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stacked bar by default status
        cat_default = pd.crosstab(df[selected_cat_feature], df['Is_Defaulter'])
        fig = px.bar(
            cat_default,
            title=f'{selected_cat_feature} by Default Status',
            barmode='stack',
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc66'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Correlations
with tab3:
    st.markdown("### 🔗 Feature Correlations")
    
    # Select numerical features for correlation
    key_numerical = ['Age', 'Estimated_Monthly_Income', 'Loan_Amount', 
                    'Outstanding_Balance', 'Overdue_Amount', 'Payments_Made', 
                    'Payments_Missed', 'Number_of_Loans', 'Debt_to_Income_Ratio',
                    'Payment_Rate', 'Loan_to_Income_Ratio', 'Default_Binary']
    
    corr_matrix = df[key_numerical].corr()
    
    # Correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation with target
    st.markdown("#### 🎯 Correlation with Default Status")
    
    target_corr = corr_matrix['Default_Binary'].drop('Default_Binary').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Positive correlations
        st.markdown("**Top Positive Correlations** (Higher = More Default Risk)")
        positive_corr = target_corr[target_corr > 0].sort_values(ascending=False)
        fig = px.bar(
            x=positive_corr.values,
            y=positive_corr.index,
            orientation='h',
            title='Features Positively Correlated with Default',
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=positive_corr.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Negative correlations
        st.markdown("**Top Negative Correlations** (Higher = Less Default Risk)")
        negative_corr = target_corr[target_corr < 0].sort_values()
        fig = px.bar(
            x=negative_corr.values,
            y=negative_corr.index,
            orientation='h',
            title='Features Negatively Correlated with Default',
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=negative_corr.values,
            color_continuous_scale='Blues_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.info(f"""
    **Key Correlation Insights:**
    - **Strongest Positive Predictor:** {target_corr.idxmax()} ({target_corr.max():.3f})
    - **Strongest Negative Predictor:** {target_corr.idxmin()} ({target_corr.min():.3f})
    - Features with correlation > 0.3 are strong predictors of default risk
    """)

# Tab 4: Default Analysis
with tab4:
    st.markdown("### 🎯 Default Rate Analysis by Categories")
    
    # Select category
    category_options = ['Location', 'Employment_Status', 'Monthly_Income_Bracket', 
                       'Loan_Product', 'Term_Type', 'Gender', 'Has_Savings_Account']
    
    selected_category = st.selectbox("Select category for default rate analysis:", category_options)
    
    # Calculate default rate
    default_rate = df.groupby(selected_category).agg({
        'Default_Binary': ['mean', 'count']
    }).round(4)
    default_rate.columns = ['Default_Rate', 'Count']
    default_rate['Default_Rate'] = (default_rate['Default_Rate'] * 100).round(2)
    default_rate = default_rate.sort_values('Default_Rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = px.bar(
            default_rate,
            y=default_rate.index,
            x='Default_Rate',
            orientation='h',
            title=f'Default Rate by {selected_category}',
            labels={'Default_Rate': 'Default Rate (%)', 'index': selected_category},
            color='Default_Rate',
            color_continuous_scale='Reds',
            text='Default_Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Data table
        st.markdown("**Default Rate Statistics:**")
        display_df = default_rate.copy()
        display_df['Default_Rate'] = display_df['Default_Rate'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Multi-dimensional analysis
    st.markdown("#### 🔍 Multi-Dimensional Default Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category1 = st.selectbox("Select first dimension:", category_options, key='cat1')
    
    with col2:
        category2 = st.selectbox("Select second dimension:", 
                                [c for c in category_options if c != category1], key='cat2')
    
    # Pivot table
    pivot_default = pd.crosstab(
        df[category1], 
        df[category2], 
        values=df['Default_Binary'], 
        aggfunc='mean'
    ) * 100
    
    # Heatmap
    fig = px.imshow(
        pivot_default,
        title=f'Default Rate (%) by {category1} and {category2}',
        color_continuous_scale='Reds',
        aspect='auto',
        labels={'color': 'Default Rate (%)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Data Quality
with tab5:
    st.markdown("### 📋 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Data Completeness")
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            st.warning(f"⚠️ Found {len(missing_df)} columns with missing values")
            st.dataframe(missing_df, hide_index=True, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows")
        else:
            st.success("✅ No duplicate rows detected!")
    
    with col2:
        st.markdown("#### 📊 Data Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': [
                'Total Records',
                'Total Features',
                'Numerical Features',
                'Categorical Features',
                'Missing Values',
                'Duplicate Rows',
                'Memory Usage'
            ],
            'Value': [
                f"{len(df):,}",
                f"{len(df.columns)}",
                f"{len(df.select_dtypes(include=[np.number]).columns)}",
                f"{len(df.select_dtypes(include=['object']).columns)}",
                f"{df.isnull().sum().sum()}",
                f"{duplicates}",
                f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Outlier detection
    st.markdown("#### 🔍 Outlier Detection")
    
    numerical_features = ['Loan_Amount', 'Estimated_Monthly_Income', 'Outstanding_Balance', 'Overdue_Amount']
    
    outlier_summary = []
    
    for col in numerical_features:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
        outlier_summary.append({
            'Feature': col,
            'Outliers': len(outliers),
            'Percentage': f"{len(outliers)/len(df)*100:.2f}%",
            'Mean': f"{mean:,.2f}",
            'Std Dev': f"{std:,.2f}"
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, hide_index=True, use_container_width=True)
    
    st.info("""
    **Outlier Detection Method:** Values beyond 3 standard deviations from the mean
    
    **Note:** Minimal outliers indicate good data quality and realistic generation
    """)

# Download processed data
st.markdown("---")
st.markdown("### 📥 Download Processed Data")

# Add engineered features to download
download_df = df.copy()
csv = download_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Data with Engineered Features",
    data=csv,
    file_name="FKC_Data_Processed.csv",
    mime="text/csv",
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>EDA Analysis Module | FKC Credit Default Prediction System</p>
    <p style="margin-top: 10px; font-size: 0.9rem;">
        Developed by: Michael Kamau Kibugu (093371) | Strathmore University, Nairobi, Kenya
    </p>
</div>
""", unsafe_allow_html=True)
