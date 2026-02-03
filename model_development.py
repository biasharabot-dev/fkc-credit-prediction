"""
FKC Credit Default Prediction - Phase 2
Data Preprocessing and Model Development
Three ML Models: Logistic Regression, Random Forest, Support Vector Machine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FKC CREDIT DEFAULT PREDICTION - ML MODEL DEVELOPMENT")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. LOADING DATA...")
print("-" * 80)
df = pd.read_csv('FKC_Credit_System_Data.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==================== DATA PREPROCESSING ====================
print("\n2. DATA PREPROCESSING")
print("-" * 80)

# Convert target to binary
df['Default_Binary'] = (df['Is_Defaulter'] == 'Yes').astype(int)
print(f"✓ Target variable created (0=No Default, 1=Default)")

# Feature Engineering
print("\n2.1 Feature Engineering...")
df['Debt_to_Income_Ratio'] = df['Outstanding_Balance'] / (df['Estimated_Monthly_Income'] + 1)
df['Payment_Rate'] = df['Payments_Made'] / (df['Payments_Made'] + df['Payments_Missed'] + 0.01)
df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Estimated_Monthly_Income'] + 1)
df['Overdue_to_Loan_Ratio'] = df['Overdue_Amount'] / (df['Total_Loan_Amount'] + 1)
df['Account_Age_Days'] = (pd.to_datetime('today') - pd.to_datetime(df['Account_Opening_Date'])).dt.days
df['Loan_Age_Days'] = (pd.to_datetime('today') - pd.to_datetime(df['Loan_Disbursement_Date'])).dt.days

print("   Created derived features:")
print("   - Debt_to_Income_Ratio")
print("   - Payment_Rate")
print("   - Loan_to_Income_Ratio")
print("   - Overdue_to_Loan_Ratio")
print("   - Account_Age_Days")
print("   - Loan_Age_Days")

# Select features for modeling
print("\n2.2 Feature Selection...")

# Numerical features
numerical_features = [
    'Age',
    'Estimated_Monthly_Income',
    'Account_Balance',
    'Loan_Amount',
    'Interest_Rate',
    'Outstanding_Balance',
    'Total_Paid',
    'Overdue_Amount',
    'Payments_Made',
    'Payments_Missed',
    'Number_of_Loans',
    'Number_of_Declined_Loans',
    # Engineered features
    'Debt_to_Income_Ratio',
    'Payment_Rate',
    'Loan_to_Income_Ratio',
    'Overdue_to_Loan_Ratio',
    'Account_Age_Days',
    'Loan_Age_Days'
]

# Categorical features
categorical_features = [
    'Gender',
    'Location',
    'Employment_Status',
    'Monthly_Income_Bracket',
    'Has_Savings_Account',
    'Loan_Product',
    'Term_Type',
    'Loan_Status'
]

print(f"   Selected {len(numerical_features)} numerical features")
print(f"   Selected {len(categorical_features)} categorical features")

# Create feature matrix
print("\n2.3 Encoding Categorical Variables...")
df_encoded = df.copy()

# Label encoding for categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

categorical_features_encoded = [col + '_encoded' for col in categorical_features]

# Combine all features
all_features = numerical_features + categorical_features_encoded

print(f"✓ Total features for modeling: {len(all_features)}")

# Create X and y
X = df_encoded[all_features]
y = df_encoded['Default_Binary']

print(f"\n   Feature matrix shape: {X.shape}")
print(f"   Target variable shape: {y.shape}")
print(f"   Class distribution: {dict(y.value_counts())}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n3. TRAIN-TEST SPLIT (80-20)")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training set class distribution: {dict(y_train.value_counts())}")
print(f"Test set class distribution: {dict(y_test.value_counts())}")

# ==================== FEATURE SCALING ====================
print("\n4. FEATURE SCALING")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")
print(f"   Mean of scaled features: {X_train_scaled.mean():.6f}")
print(f"   Std of scaled features: {X_train_scaled.std():.6f}")

# ==================== MODEL 1: LOGISTIC REGRESSION ====================
print("\n5. MODEL 1: LOGISTIC REGRESSION")
print("="*80)

print("\n5.1 Training Baseline Logistic Regression...")
lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr_baseline = lr_baseline.predict(X_test_scaled)
y_pred_lr_baseline_proba = lr_baseline.predict_proba(X_test_scaled)[:, 1]

# Evaluation
lr_baseline_results = {
    'Model': 'Logistic Regression (Baseline)',
    'Accuracy': accuracy_score(y_test, y_pred_lr_baseline),
    'Precision': precision_score(y_test, y_pred_lr_baseline),
    'Recall': recall_score(y_test, y_pred_lr_baseline),
    'F1_Score': f1_score(y_test, y_pred_lr_baseline),
    'ROC_AUC': roc_auc_score(y_test, y_pred_lr_baseline_proba)
}

print("\n   Baseline Performance:")
for metric, value in lr_baseline_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

print("\n5.2 Hyperparameter Tuning with GridSearchCV...")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    lr_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_

print(f"   Best parameters: {lr_grid.best_params_}")
print(f"   Best CV ROC-AUC: {lr_grid.best_score_:.4f}")

# Final predictions with tuned model
y_pred_lr = lr_best.predict(X_test_scaled)
y_pred_lr_proba = lr_best.predict_proba(X_test_scaled)[:, 1]

lr_results = {
    'Model': 'Logistic Regression (Tuned)',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1_Score': f1_score(y_test, y_pred_lr),
    'ROC_AUC': roc_auc_score(y_test, y_pred_lr_proba)
}

print("\n   Tuned Model Performance:")
for metric, value in lr_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\n   Confusion Matrix:")
print(f"   {cm_lr}")

# Classification Report
print("\n   Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Default', 'Default']))

# ==================== MODEL 2: RANDOM FOREST ====================
print("\n6. MODEL 2: RANDOM FOREST")
print("="*80)

print("\n6.1 Training Baseline Random Forest...")
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf_baseline = rf_baseline.predict(X_test_scaled)
y_pred_rf_baseline_proba = rf_baseline.predict_proba(X_test_scaled)[:, 1]

# Evaluation
rf_baseline_results = {
    'Model': 'Random Forest (Baseline)',
    'Accuracy': accuracy_score(y_test, y_pred_rf_baseline),
    'Precision': precision_score(y_test, y_pred_rf_baseline),
    'Recall': recall_score(y_test, y_pred_rf_baseline),
    'F1_Score': f1_score(y_test, y_pred_rf_baseline),
    'ROC_AUC': roc_auc_score(y_test, y_pred_rf_baseline_proba)
}

print("\n   Baseline Performance:")
for metric, value in rf_baseline_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

print("\n6.2 Hyperparameter Tuning with GridSearchCV...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_

print(f"   Best parameters: {rf_grid.best_params_}")
print(f"   Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

# Final predictions with tuned model
y_pred_rf = rf_best.predict(X_test_scaled)
y_pred_rf_proba = rf_best.predict_proba(X_test_scaled)[:, 1]

rf_results = {
    'Model': 'Random Forest (Tuned)',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1_Score': f1_score(y_test, y_pred_rf),
    'ROC_AUC': roc_auc_score(y_test, y_pred_rf_proba)
}

print("\n   Tuned Model Performance:")
for metric, value in rf_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\n   Confusion Matrix:")
print(f"   {cm_rf}")

# Classification Report
print("\n   Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Default', 'Default']))

# Feature Importance
print("\n6.3 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ==================== MODEL 3: SUPPORT VECTOR MACHINE ====================
print("\n7. MODEL 3: SUPPORT VECTOR MACHINE (SVM)")
print("="*80)

print("\n7.1 Training Baseline SVM (Linear Kernel)...")
svm_baseline = SVC(kernel='linear', random_state=42, probability=True)
svm_baseline.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm_baseline = svm_baseline.predict(X_test_scaled)
y_pred_svm_baseline_proba = svm_baseline.predict_proba(X_test_scaled)[:, 1]

# Evaluation
svm_baseline_results = {
    'Model': 'SVM (Linear Baseline)',
    'Accuracy': accuracy_score(y_test, y_pred_svm_baseline),
    'Precision': precision_score(y_test, y_pred_svm_baseline),
    'Recall': recall_score(y_test, y_pred_svm_baseline),
    'F1_Score': f1_score(y_test, y_pred_svm_baseline),
    'ROC_AUC': roc_auc_score(y_test, y_pred_svm_baseline_proba)
}

print("\n   Baseline Performance (Linear Kernel):")
for metric, value in svm_baseline_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

print("\n7.2 Hyperparameter Tuning with GridSearchCV...")
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(
    SVC(random_state=42, probability=True),
    svm_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_

print(f"   Best parameters: {svm_grid.best_params_}")
print(f"   Best CV ROC-AUC: {svm_grid.best_score_:.4f}")

# Final predictions with tuned model
y_pred_svm = svm_best.predict(X_test_scaled)
y_pred_svm_proba = svm_best.predict_proba(X_test_scaled)[:, 1]

svm_results = {
    'Model': 'SVM (Tuned)',
    'Accuracy': accuracy_score(y_test, y_pred_svm),
    'Precision': precision_score(y_test, y_pred_svm),
    'Recall': recall_score(y_test, y_pred_svm),
    'F1_Score': f1_score(y_test, y_pred_svm),
    'ROC_AUC': roc_auc_score(y_test, y_pred_svm_proba)
}

print("\n   Tuned Model Performance:")
for metric, value in svm_results.items():
    if metric != 'Model':
        print(f"   {metric}: {value:.4f}")

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\n   Confusion Matrix:")
print(f"   {cm_svm}")

# Classification Report
print("\n   Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['No Default', 'Default']))

# ==================== MODEL COMPARISON ====================
print("\n8. MODEL COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame([lr_results, rf_results, svm_results])
comparison_df = comparison_df.round(4)

print("\nPerformance Comparison Table:")
print(comparison_df.to_string(index=False))

# Determine best model
best_model_idx = comparison_df['ROC_AUC'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_model_roc = comparison_df.loc[best_model_idx, 'ROC_AUC']

print(f"\n{'*'*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"ROC-AUC Score: {best_model_roc:.4f}")
print(f"{'*'*80}")

# ==================== CROSS-VALIDATION ====================
print("\n9. CROSS-VALIDATION ANALYSIS (5-Fold)")
print("="*80)

# Logistic Regression
lr_cv_scores = cross_val_score(lr_best, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\nLogistic Regression CV Scores: {lr_cv_scores}")
print(f"Mean CV ROC-AUC: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")

# Random Forest
rf_cv_scores = cross_val_score(rf_best, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\nRandom Forest CV Scores: {rf_cv_scores}")
print(f"Mean CV ROC-AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# SVM
svm_cv_scores = cross_val_score(svm_best, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\nSVM CV Scores: {svm_cv_scores}")
print(f"Mean CV ROC-AUC: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")

# ==================== SAVE RESULTS ====================
print("\n10. SAVING RESULTS")
print("="*80)

# Save comparison table
comparison_df.to_csv('Model_Comparison_Results.csv', index=False)
print("✓ Model comparison saved to 'Model_Comparison_Results.csv'")

# Save feature importance
feature_importance.to_csv('Feature_Importance_RandomForest.csv', index=False)
print("✓ Feature importance saved to 'Feature_Importance_RandomForest.csv'")

# Create results summary
results_summary = {
    'Training_Samples': len(X_train),
    'Test_Samples': len(X_test),
    'Number_of_Features': len(all_features),
    'Best_Model': best_model_name,
    'Best_ROC_AUC': best_model_roc,
    'LR_Accuracy': lr_results['Accuracy'],
    'RF_Accuracy': rf_results['Accuracy'],
    'SVM_Accuracy': svm_results['Accuracy'],
    'LR_ROC_AUC': lr_results['ROC_AUC'],
    'RF_ROC_AUC': rf_results['ROC_AUC'],
    'SVM_ROC_AUC': svm_results['ROC_AUC']
}

results_summary_df = pd.DataFrame([results_summary])
results_summary_df.to_csv('Results_Summary.csv', index=False)
print("✓ Results summary saved to 'Results_Summary.csv'")

print("\n" + "="*80)
print("MODEL DEVELOPMENT COMPLETE!")
print("="*80)
print("\nAll three models have been successfully trained and evaluated.")
print(f"Best performing model: {best_model_name}")
print(f"ROC-AUC Score: {best_model_roc:.4f}")