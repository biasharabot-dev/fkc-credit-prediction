"""
FKC Credit Default Prediction System
Page 3: Model Training
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import time

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Training")
st.markdown("Train and compare multiple ML models for credit default prediction")

st.markdown("---")

# Load data
data_path = Path("data/FKC_Credit_System_Data.csv")

if not data_path.exists():
    st.warning("⚠️ No data file found. Please generate data first from the Data Generation page.")
    st.stop()

# Load the data
df = pd.read_csv(data_path)

st.success(f"✅ Loaded {len(df):,} customer records")

# Model Configuration
st.markdown("### ⚙️ Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    
with col2:
    cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1)

with col3:
    random_state = st.number_input("Random State", 0, 100, 42)

st.markdown("---")

# Model selection
st.markdown("### 🎯 Select Models to Train")

col1, col2, col3 = st.columns(3)

with col1:
    train_lr = st.checkbox("Logistic Regression", value=True)
    
with col2:
    train_rf = st.checkbox("Random Forest", value=True)
    
with col3:
    train_svm = st.checkbox("Support Vector Machine", value=True)

if not any([train_lr, train_rf, train_svm]):
    st.warning("⚠️ Please select at least one model to train")
    st.stop()

st.markdown("---")

# Training button
if st.button("🚀 Start Training", type="primary", use_container_width=True):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Data Preprocessing
        status_text.text("📊 Preprocessing data...")
        progress_bar.progress(5)
        
        # Create binary target
        df['Default_Binary'] = (df['Is_Defaulter'] == 'Yes').astype(int)
        
        # Feature Engineering
        df['Debt_to_Income_Ratio'] = df['Outstanding_Balance'] / (df['Estimated_Monthly_Income'] + 1)
        df['Payment_Rate'] = df['Payments_Made'] / (df['Payments_Made'] + df['Payments_Missed'] + 0.01)
        df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Estimated_Monthly_Income'] + 1)
        df['Overdue_to_Loan_Ratio'] = df['Overdue_Amount'] / (df['Total_Loan_Amount'] + 1)
        df['Account_Age_Days'] = (pd.to_datetime('today') - pd.to_datetime(df['Account_Opening_Date'])).dt.days
        df['Loan_Age_Days'] = (pd.to_datetime('today') - pd.to_datetime(df['Loan_Disbursement_Date'])).dt.days
        
        progress_bar.progress(10)
        
        # Feature Selection
        numerical_features = [
            'Age', 'Estimated_Monthly_Income', 'Account_Balance', 'Loan_Amount',
            'Interest_Rate', 'Outstanding_Balance', 'Total_Paid', 'Overdue_Amount',
            'Payments_Made', 'Payments_Missed', 'Number_of_Loans', 'Number_of_Declined_Loans',
            'Debt_to_Income_Ratio', 'Payment_Rate', 'Loan_to_Income_Ratio',
            'Overdue_to_Loan_Ratio', 'Account_Age_Days', 'Loan_Age_Days'
        ]
        
        categorical_features = [
            'Gender', 'Location', 'Employment_Status', 'Monthly_Income_Bracket',
            'Has_Savings_Account', 'Loan_Product', 'Term_Type', 'Loan_Status'
        ]
        
        # Encode categorical variables
        status_text.text("🔤 Encoding categorical variables...")
        df_encoded = df.copy()
        label_encoders = {}
        
        for col in categorical_features:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
        
        categorical_features_encoded = [col + '_encoded' for col in categorical_features]
        all_features = numerical_features + categorical_features_encoded
        
        progress_bar.progress(15)
        
        # Create X and y
        X = df_encoded[all_features]
        y = df_encoded['Default_Binary']
        
        # Train-test split
        status_text.text("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        progress_bar.progress(20)
        
        # Feature Scaling
        status_text.text("📏 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        progress_bar.progress(25)
        
        # Store results
        results = []
        models_trained = {}
        
        # Train Logistic Regression
        if train_lr:
            status_text.text("🔵 Training Logistic Regression...")
            
            lr_param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            lr_grid = GridSearchCV(
                LogisticRegression(random_state=random_state, max_iter=1000),
                lr_param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            lr_grid.fit(X_train_scaled, y_train)
            lr_best = lr_grid.best_estimator_
            
            y_pred_lr = lr_best.predict(X_test_scaled)
            y_pred_lr_proba = lr_best.predict_proba(X_test_scaled)[:, 1]
            
            results.append({
                'Model': 'Logistic Regression',
                'Accuracy': accuracy_score(y_test, y_pred_lr),
                'Precision': precision_score(y_test, y_pred_lr),
                'Recall': recall_score(y_test, y_pred_lr),
                'F1_Score': f1_score(y_test, y_pred_lr),
                'ROC_AUC': roc_auc_score(y_test, y_pred_lr_proba),
                'Best_Params': str(lr_grid.best_params_)
            })
            
            models_trained['Logistic Regression'] = {
                'model': lr_best,
                'predictions': y_pred_lr,
                'probabilities': y_pred_lr_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
            }
            
            progress_bar.progress(45)
        
        # Train Random Forest
        if train_rf:
            status_text.text("🌲 Training Random Forest...")
            
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                rf_param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            rf_grid.fit(X_train_scaled, y_train)
            rf_best = rf_grid.best_estimator_
            
            y_pred_rf = rf_best.predict(X_test_scaled)
            y_pred_rf_proba = rf_best.predict_proba(X_test_scaled)[:, 1]
            
            results.append({
                'Model': 'Random Forest',
                'Accuracy': accuracy_score(y_test, y_pred_rf),
                'Precision': precision_score(y_test, y_pred_rf),
                'Recall': recall_score(y_test, y_pred_rf),
                'F1_Score': f1_score(y_test, y_pred_rf),
                'ROC_AUC': roc_auc_score(y_test, y_pred_rf_proba),
                'Best_Params': str(rf_grid.best_params_)
            })
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': rf_best.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            models_trained['Random Forest'] = {
                'model': rf_best,
                'predictions': y_pred_rf,
                'probabilities': y_pred_rf_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
                'feature_importance': feature_importance
            }
            
            progress_bar.progress(70)
        
        # Train SVM
        if train_svm:
            status_text.text("🔴 Training Support Vector Machine...")
            
            svm_param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            
            svm_grid = GridSearchCV(
                SVC(random_state=random_state, probability=True),
                svm_param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            svm_grid.fit(X_train_scaled, y_train)
            svm_best = svm_grid.best_estimator_
            
            y_pred_svm = svm_best.predict(X_test_scaled)
            y_pred_svm_proba = svm_best.predict_proba(X_test_scaled)[:, 1]
            
            results.append({
                'Model': 'SVM',
                'Accuracy': accuracy_score(y_test, y_pred_svm),
                'Precision': precision_score(y_test, y_pred_svm),
                'Recall': recall_score(y_test, y_pred_svm),
                'F1_Score': f1_score(y_test, y_pred_svm),
                'ROC_AUC': roc_auc_score(y_test, y_pred_svm_proba),
                'Best_Params': str(svm_grid.best_params_)
            })
            
            models_trained['SVM'] = {
                'model': svm_best,
                'predictions': y_pred_svm,
                'probabilities': y_pred_svm_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred_svm)
            }
            
            progress_bar.progress(90)
        
        # Save best model
        status_text.text("💾 Saving models...")
        
        results_df = pd.DataFrame(results)
        best_model_idx = results_df['ROC_AUC'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Model']
        best_model = models_trained[best_model_name]['model']
        
        # Save to file
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(best_model, models_dir / "best_model.pkl")
        joblib.dump(scaler, models_dir / "scaler.pkl")
        joblib.dump(label_encoders, models_dir / "label_encoders.pkl")
        joblib.dump(all_features, models_dir / "feature_names.pkl")
        
        progress_bar.progress(100)
        status_text.empty()
        
        st.success(f"✅ Training Complete! Best Model: **{best_model_name}**")
        
        # Display Results
        st.markdown("---")
        st.markdown("### 📊 Model Comparison Results")
        
        # Format results
        display_results = results_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
            display_results[col] = display_results[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_results, use_container_width=True, hide_index=True)
        
        # Highlight best model
        st.info(f"""
        **🏆 Best Model: {best_model_name}**
        - ROC-AUC Score: {results_df.loc[best_model_idx, 'ROC_AUC']:.4f}
        - Accuracy: {results_df.loc[best_model_idx, 'Accuracy']:.4f}
        - Best Parameters: {results_df.loc[best_model_idx, 'Best_Params']}
        """)
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Performance Visualizations")
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of metrics
            metrics_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].melt(
                id_vars='Model', var_name='Metric', value_name='Score'
            )
            
            fig = px.bar(
                metrics_df,
                x='Metric',
                y='Score',
                color='Model',
                barmode='group',
                title='Model Performance Comparison',
                labels={'Score': 'Score', 'Metric': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC-AUC comparison
            fig = px.bar(
                results_df,
                x='Model',
                y='ROC_AUC',
                title='ROC-AUC Score Comparison',
                color='ROC_AUC',
                color_continuous_scale='Blues',
                text='ROC_AUC'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices
        st.markdown("---")
        st.markdown("### 🎯 Confusion Matrices")
        
        cols = st.columns(len(models_trained))
        
        for idx, (model_name, model_data) in enumerate(models_trained.items()):
            with cols[idx]:
                cm = model_data['confusion_matrix']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Default', 'Default'],
                    y=['No Default', 'Default'],
                    title=f'{model_name}',
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance (if Random Forest was trained)
        if train_rf and 'Random Forest' in models_trained:
            st.markdown("---")
            st.markdown("### 🌟 Feature Importance (Random Forest)")
            
            feature_imp = models_trained['Random Forest']['feature_importance'].head(15)
            
            fig = px.bar(
                feature_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Store in session state
        st.session_state['models_trained'] = True
        st.session_state['best_model_name'] = best_model_name
        st.session_state['results_df'] = results_df
        
    except Exception as e:
        st.error(f"❌ Error during training: {e}")
        import traceback
        st.code(traceback.format_exc())

# Check if models exist
st.markdown("---")
st.markdown("### 💾 Saved Models")

models_dir = Path("models")

if (models_dir / "best_model.pkl").exists():
    st.success("✅ Trained model available")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Best Model"):
            model_data = open(models_dir / "best_model.pkl", "rb").read()
            st.download_button(
                label="Download Model File",
                data=model_data,
                file_name="fkc_best_model.pkl",
                mime="application/octet-stream"
            )
    
    with col2:
        if st.button("📥 Download Scaler"):
            scaler_data = open(models_dir / "scaler.pkl", "rb").read()
            st.download_button(
                label="Download Scaler File",
                data=scaler_data,
                file_name="fkc_scaler.pkl",
                mime="application/octet-stream"
            )
    
    with col3:
        if st.button("📥 Download Encoders"):
            encoder_data = open(models_dir / "label_encoders.pkl", "rb").read()
            st.download_button(
                label="Download Encoders File",
                data=encoder_data,
                file_name="fkc_encoders.pkl",
                mime="application/octet-stream"
            )

else:
    st.info("ℹ️ No trained models found. Train models using the button above.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Model Training Module | FKC Credit Default Prediction System</p>
    <p style="margin-top: 10px; font-size: 0.9rem;">
        Developed by: Michael Kamau Kibugu (093371) | Strathmore University, Nairobi, Kenya
    </p>
</div>
""", unsafe_allow_html=True)
