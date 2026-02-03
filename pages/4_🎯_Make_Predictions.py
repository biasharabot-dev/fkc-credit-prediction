"""
FKC Credit Default Prediction System
Page 4: Make Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
from datetime import datetime, timedelta

st.set_page_config(page_title="Make Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Credit Default Risk Prediction")
st.markdown("Enter customer details to predict default probability")

st.markdown("---")

# Check if model exists
models_dir = Path("models")
model_path = models_dir / "best_model.pkl"
scaler_path = models_dir / "scaler.pkl"
encoders_path = models_dir / "label_encoders.pkl"
features_path = models_dir / "feature_names.pkl"

if not all([model_path.exists(), scaler_path.exists(), encoders_path.exists(), features_path.exists()]):
    st.warning("⚠️ No trained model found. Please train a model first from the Model Training page.")
    st.stop()

# Load model and preprocessing objects
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    feature_names = joblib.load(features_path)
    st.success(f"✅ Model loaded successfully: **{type(model).__name__}**")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.markdown("---")

# Input method selection
input_method = st.radio(
    "Select Input Method:",
    ["📝 Manual Entry", "📊 Batch Prediction (CSV Upload)"],
    horizontal=True
)

st.markdown("---")

if input_method == "📝 Manual Entry":
    st.markdown("### 📝 Enter Customer Information")
    
    # Create input form
    with st.form("prediction_form"):
        # Personal Information
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            location = st.selectbox("Location", [
                "Nairobi", "Kayole", "Kamulu", "Mombasa Road", "Mwea"
            ])
            employment_status = st.selectbox("Employment Status", [
                "Formally employed (permanent)",
                "Formally employed (contract)",
                "Self-employed/Business owner",
                "Casual laborer"
            ])
        
        with col3:
            income_bracket = st.selectbox("Monthly Income Bracket", [
                "Below 10,000",
                "10,000 - 20,000",
                "20,001 - 30,000",
                "30,001 - 50,000",
                "50,001 - 100,000",
                "Above 100,000"
            ])
            has_savings = st.selectbox("Has Savings Account", ["Yes", "No"])
        
        # Financial Information
        st.markdown("#### 💰 Financial Information")
        col1, col2, col3 = st.columns(3)
        
        # Income midpoints for estimation
        income_midpoints = {
            'Below 10,000': 7500,
            '10,000 - 20,000': 15000,
            '20,001 - 30,000': 25000,
            '30,001 - 50,000': 40000,
            '50,001 - 100,000': 75000,
            'Above 100,000': 150000
        }
        estimated_income = income_midpoints[income_bracket]
        
        with col1:
            account_balance = st.number_input(
                "Account Balance (KES)", 
                min_value=0, 
                max_value=1000000, 
                value=int(estimated_income * 0.5),
                step=1000
            )
        
        with col2:
            num_loans = st.number_input("Number of Existing Loans", min_value=1, max_value=10, value=1)
            num_declined = st.number_input("Number of Declined Loans", min_value=0, max_value=5, value=0)
        
        with col3:
            account_age_days = st.number_input(
                "Account Age (Days)", 
                min_value=30, 
                max_value=1095, 
                value=365,
                help="How long has the customer been with FKC?"
            )
        
        # Loan Information
        st.markdown("#### 🏦 Loan Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_product = st.selectbox("Loan Product", [
                "School Fees Loan",
                "Utility Loan",
                "Cash Loan",
                "Business Capital Loan",
                "Emergency Loan"
            ])
            
            # Interest rates by product
            interest_rates = {
                "School Fees Loan": 14.5,
                "Utility Loan": 16.0,
                "Cash Loan": 18.5,
                "Business Capital Loan": 19.5,
                "Emergency Loan": 22.0
            }
            interest_rate = interest_rates[loan_product]
            st.info(f"Interest Rate: {interest_rate}%")
        
        with col2:
            loan_amount = st.number_input(
                "Loan Amount (KES)", 
                min_value=10000, 
                max_value=500000, 
                value=50000,
                step=5000
            )
            term_type = st.selectbox("Term Type", ["monthly", "weekly"])
        
        with col3:
            if term_type == "monthly":
                term_months = st.selectbox("Loan Term (Months)", [3, 6, 9, 12])
                term_weeks = None
            else:
                term_weeks = st.selectbox("Loan Term (Weeks)", [12, 24, 36, 48])
                term_months = None
            
            loan_status = st.selectbox("Current Loan Status", [
                "Active", "Overdue", "Completed", "Defaulted"
            ])
        
        # Payment Information
        st.markdown("#### 💳 Payment Information")
        col1, col2, col3 = st.columns(3)
        
        total_loan_amount = loan_amount * (1 + interest_rate/100)
        
        with col1:
            loan_age_days = st.number_input(
                "Loan Age (Days)", 
                min_value=1, 
                max_value=365, 
                value=90,
                help="How long ago was the loan disbursed?"
            )
        
        with col2:
            payments_made = st.number_input("Payments Made", min_value=0, max_value=50, value=5)
            payments_missed = st.number_input("Payments Missed", min_value=0, max_value=20, value=0)
        
        with col3:
            total_paid = st.number_input(
                "Total Amount Paid (KES)", 
                min_value=0, 
                max_value=int(total_loan_amount), 
                value=int(total_loan_amount * 0.3),
                step=1000
            )
            overdue_amount = st.number_input(
                "Overdue Amount (KES)", 
                min_value=0, 
                max_value=int(total_loan_amount), 
                value=0,
                step=1000
            )
        
        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict Default Risk", use_container_width=True, type="primary")
    
    if submitted:
        try:
            # Calculate derived features
            outstanding_balance = total_loan_amount - total_paid
            debt_to_income_ratio = outstanding_balance / (estimated_income + 1)
            payment_rate = payments_made / (payments_made + payments_missed + 0.01)
            loan_to_income_ratio = loan_amount / (estimated_income + 1)
            overdue_to_loan_ratio = overdue_amount / (total_loan_amount + 1)
            
            # Create feature dictionary
            input_data = {
                'Age': age,
                'Estimated_Monthly_Income': estimated_income,
                'Account_Balance': account_balance,
                'Loan_Amount': loan_amount,
                'Interest_Rate': interest_rate,
                'Outstanding_Balance': outstanding_balance,
                'Total_Paid': total_paid,
                'Overdue_Amount': overdue_amount,
                'Payments_Made': payments_made,
                'Payments_Missed': payments_missed,
                'Number_of_Loans': num_loans,
                'Number_of_Declined_Loans': num_declined,
                'Debt_to_Income_Ratio': debt_to_income_ratio,
                'Payment_Rate': payment_rate,
                'Loan_to_Income_Ratio': loan_to_income_ratio,
                'Overdue_to_Loan_Ratio': overdue_to_loan_ratio,
                'Account_Age_Days': account_age_days,
                'Loan_Age_Days': loan_age_days
            }
            
            # Encode categorical features
            categorical_mapping = {
                'Gender': gender,
                'Location': location,
                'Employment_Status': employment_status,
                'Monthly_Income_Bracket': income_bracket,
                'Has_Savings_Account': has_savings,
                'Loan_Product': loan_product,
                'Term_Type': term_type,
                'Loan_Status': loan_status
            }
            
            for col, value in categorical_mapping.items():
                encoded_value = label_encoders[col].transform([value])[0]
                input_data[col + '_encoded'] = encoded_value
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names]
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            default_probability = probability[1] * 100
            no_default_probability = probability[0] * 100
            
            # Display Results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Risk level determination
            if default_probability < 30:
                risk_level = "LOW"
                risk_color = "green"
                risk_emoji = "✅"
                recommendation = "**APPROVE** - Low risk customer. Proceed with loan approval."
            elif default_probability < 60:
                risk_level = "MEDIUM"
                risk_color = "orange"
                risk_emoji = "⚠️"
                recommendation = "**REVIEW** - Medium risk. Additional verification recommended."
            else:
                risk_level = "HIGH"
                risk_color = "red"
                risk_emoji = "❌"
                recommendation = "**REJECT** - High risk customer. Loan approval not recommended."
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            # Determine background gradients based on risk level
            if risk_level == "LOW":
                bg_gradient = "linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"
            elif risk_level == "MEDIUM":
                bg_gradient = "linear-gradient(135deg, #f2994a 0%, #f2c94c 100%)"
            else:
                bg_gradient = "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)"
            
            with col1:
                st.markdown(f"""
                <div style="background: {bg_gradient}; 
                            padding: 25px; border-radius: 15px; text-align: center; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">Prediction</h3>
                    <h1 style="color: white; margin: 0; font-size: 2rem;">{risk_emoji} {'DEFAULT' if prediction == 1 else 'NO DEFAULT'}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: {bg_gradient}; 
                            padding: 25px; border-radius: 15px; text-align: center; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">Default Probability</h3>
                    <h1 style="color: white; margin: 0; font-size: 2rem;">{default_probability:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: {bg_gradient}; 
                            padding: 25px; border-radius: 15px; text-align: center; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">Risk Level</h3>
                    <h1 style="color: white; margin: 0; font-size: 2rem;">{risk_emoji} {risk_level}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("---")
            st.markdown("### 📊 Risk Assessment Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=default_probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk Probability (%)", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'lightyellow'},
                        {'range': [60, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            
            if risk_level == "LOW":
                st.success(recommendation)
            elif risk_level == "MEDIUM":
                st.warning(recommendation)
            else:
                st.error(recommendation)
            
            # Key Risk Factors
            st.markdown("---")
            st.markdown("### 🔍 Key Risk Factors")
            
            risk_factors = []
            
            if payments_missed > 0:
                risk_factors.append(f"❌ **Missed Payments:** {payments_missed} payments missed")
            
            if overdue_amount > 0:
                risk_factors.append(f"❌ **Overdue Amount:** KES {overdue_amount:,} outstanding")
            
            if payment_rate < 0.8:
                risk_factors.append(f"❌ **Low Payment Rate:** {payment_rate*100:.1f}% payment completion")
            
            if debt_to_income_ratio > 0.5:
                risk_factors.append(f"❌ **High Debt-to-Income:** {debt_to_income_ratio:.2f} ratio")
            
            if loan_status in ["Overdue", "Defaulted"]:
                risk_factors.append(f"❌ **Loan Status:** Current loan is {loan_status}")
            
            if num_declined > 0:
                risk_factors.append(f"⚠️ **Previous Declines:** {num_declined} loan(s) previously declined")
            
            # Positive factors
            positive_factors = []
            
            if payment_rate >= 0.9:
                positive_factors.append(f"✅ **Good Payment Rate:** {payment_rate*100:.1f}% payment completion")
            
            if has_savings == "Yes":
                positive_factors.append("✅ **Has Savings Account:** Customer has savings")
            
            if payments_missed == 0:
                positive_factors.append("✅ **No Missed Payments:** Perfect payment history")
            
            if debt_to_income_ratio < 0.3:
                positive_factors.append(f"✅ **Low Debt-to-Income:** {debt_to_income_ratio:.2f} ratio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if risk_factors:
                    st.markdown("**⚠️ Risk Indicators:**")
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("✅ No major risk indicators found")
            
            with col2:
                if positive_factors:
                    st.markdown("**✅ Positive Indicators:**")
                    for factor in positive_factors:
                        st.markdown(factor)
                else:
                    st.info("No strong positive indicators")
            
            # Customer Summary
            st.markdown("---")
            st.markdown("### 📋 Customer Summary")
            
            summary_data = {
                "Category": [
                    "Personal", "Personal", "Personal",
                    "Financial", "Financial", "Financial",
                    "Loan", "Loan", "Loan",
                    "Payment", "Payment", "Payment"
                ],
                "Attribute": [
                    "Age", "Location", "Employment",
                    "Monthly Income", "Account Balance", "Savings Account",
                    "Loan Product", "Loan Amount", "Interest Rate",
                    "Payments Made", "Payments Missed", "Payment Rate"
                ],
                "Value": [
                    f"{age} years", location, employment_status,
                    f"KES {estimated_income:,}", f"KES {account_balance:,}", has_savings,
                    loan_product, f"KES {loan_amount:,}", f"{interest_rate}%",
                    payments_made, payments_missed, f"{payment_rate*100:.1f}%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    # Batch prediction
    st.markdown("### 📊 Batch Prediction from CSV")
    
    st.info("""
    **Upload a CSV file with customer data for batch predictions.**
    
    The CSV should contain the same features as the training data.
    Download the sample template below to see the required format.
    """)
    
    # Sample template
    if st.button("📥 Download Sample Template"):
        sample_data = {
            'Age': [35, 28, 45],
            'Gender': ['Male', 'Female', 'Male'],
            'Location': ['Nairobi', 'Kayole', 'Kamulu'],
            'Employment_Status': ['Formally employed (permanent)', 'Self-employed/Business owner', 'Casual laborer'],
            'Monthly_Income_Bracket': ['30,001 - 50,000', '20,001 - 30,000', '10,000 - 20,000'],
            'Has_Savings_Account': ['Yes', 'No', 'Yes'],
            'Account_Balance': [50000, 10000, 5000],
            'Number_of_Loans': [1, 2, 1],
            'Number_of_Declined_Loans': [0, 1, 0],
            'Account_Age_Days': [365, 180, 730],
            'Loan_Product': ['Cash Loan', 'Emergency Loan', 'Utility Loan'],
            'Loan_Amount': [50000, 30000, 20000],
            'Term_Type': ['monthly', 'weekly', 'monthly'],
            'Loan_Status': ['Active', 'Overdue', 'Active'],
            'Loan_Age_Days': [90, 120, 60],
            'Payments_Made': [5, 3, 4],
            'Payments_Missed': [0, 2, 0],
            'Total_Paid': [15000, 8000, 12000],
            'Overdue_Amount': [0, 5000, 0]
        }
        
        sample_df = pd.DataFrame(sample_data)
        csv = sample_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Template CSV",
            data=csv,
            file_name="fkc_prediction_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} records for prediction")
            
            st.markdown("**Preview of uploaded data:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔮 Run Batch Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    # Process batch predictions (simplified version)
                    st.info("Batch prediction feature coming soon! For now, please use manual entry.")
                    
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Prediction Module | FKC Credit Default Prediction System</p>
    <p style="margin-top: 10px; font-size: 0.9rem;">
        Developed by: Michael Kamau Kibugu (093371) | Strathmore University, Nairobi, Kenya
    </p>
</div>
""", unsafe_allow_html=True)
