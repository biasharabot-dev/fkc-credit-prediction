"""
FKC Credit Default Prediction System
Page 1: Data Generation
"""

import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path to import data generator
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Data Generation", page_icon="📊", layout="wide")

st.title("📊 Data Generation")
st.markdown("Generate realistic customer credit data for FKC Credit System")

st.markdown("---")

# Import the data generation functions
try:
    from fkc_data_generator import (
        generate_customer_record, TOTAL_RECORDS, DEFAULTERS, 
        NON_DEFAULTERS, OPERATING_LOCATIONS, LOAN_PRODUCTS
    )
    generator_available = True
except Exception as e:
    st.error(f"Error importing data generator: {e}")
    generator_available = False

# Configuration Display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{2500:,}")
    st.metric("Defaulters", f"{750:,}", "30%")

with col2:
    st.metric("Non-Defaulters", f"{1750:,}", "70%")
    st.metric("Operating Locations", "5")

with col3:
    st.metric("Loan Products", "5")
    st.metric("Customer Profiles", "5")

st.markdown("---")

# System Configuration
with st.expander("📋 View System Configuration"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Operating Locations")
        locations_df = pd.DataFrame([
            {"Location": "Nairobi", "Distribution": "50%"},
            {"Location": "Kayole", "Distribution": "20%"},
            {"Location": "Kamulu", "Distribution": "15%"},
            {"Location": "Mombasa Road", "Distribution": "10%"},
            {"Location": "Mwea", "Distribution": "5%"}
        ])
        st.dataframe(locations_df, hide_index=True, use_container_width=True)
        
        st.markdown("#### 💰 Loan Products")
        products_df = pd.DataFrame([
            {"Product": "School Fees Loan", "Interest Rate": "14.5%"},
            {"Product": "Utility Loan", "Interest Rate": "16.0%"},
            {"Product": "Cash Loan", "Interest Rate": "18.5%"},
            {"Product": "Business Capital Loan", "Interest Rate": "19.5%"},
            {"Product": "Emergency Loan", "Interest Rate": "22.0%"}
        ])
        st.dataframe(products_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### 👥 Customer Profiles")
        profiles_df = pd.DataFrame([
            {"Profile": "Stable Professional", "Default Rate": "15%", "Weight": "25%"},
            {"Profile": "Young Entrepreneur", "Default Rate": "25%", "Weight": "20%"},
            {"Profile": "Hustler/Casual Worker", "Default Rate": "45%", "Weight": "25%"},
            {"Profile": "Overburdened Family Person", "Default Rate": "40%", "Weight": "20%"},
            {"Profile": "Entry Level Worker", "Default Rate": "20%", "Weight": "10%"}
        ])
        st.dataframe(profiles_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Data Generation Section
st.markdown("### 🚀 Generate Data")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Click the button below to generate 2,500 realistic customer records. The process includes:
    
    - ✅ Customer demographics (name, age, gender, location)
    - ✅ Financial information (income, savings, account balance)
    - ✅ Loan details (product, amount, interest rate, payment history)
    - ✅ Default status (30% defaulters, 70% non-defaulters)
    - ✅ Privacy protection (masked IDs, phone numbers, emails)
    
    **Note:** This process may take 30-60 seconds to complete.
    """)

with col2:
    st.info("💡 **Tip**: Generated data will be saved to `data/FKC_Credit_System_Data.csv`")

st.markdown("---")

# Option to upload existing CSV
st.markdown("### 📤 Or Upload Existing Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file with customer data", 
    type=['csv'],
    help="Upload an existing FKC customer data CSV file"
)

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df_upload):,} records from uploaded file")
        
        # Save to data folder
        output_path = Path("data/FKC_Credit_System_Data.csv")
        output_path.parent.mkdir(exist_ok=True)
        df_upload.to_csv(output_path, index=False)
        
        st.info("📁 Data saved to `data/FKC_Credit_System_Data.csv`")
        
        # Show preview
        st.markdown("**Preview:**")
        display_cols = ['Customer_ID', 'Full_Name', 'Age', 'Gender', 'Location', 
                       'Loan_Product', 'Loan_Amount', 'Is_Defaulter']
        available_cols = [col for col in display_cols if col in df_upload.columns]
        st.dataframe(df_upload[available_cols].head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error reading uploaded file: {e}")

st.markdown("---")
st.markdown("### 🎲 Or Generate New Data")

# Generate button
if st.button("🎲 Generate Customer Data", type="primary", use_container_width=True):
    if not generator_available:
        st.error("❌ Data generator not available. Please check the import.")
    else:
        with st.spinner("Generating 2,500 customer records... Please wait..."):
            try:
                # Initialize (fix Faker locale)
                fake = Faker('en_US')  # Changed from en_KE to en_US
                np.random.seed(42)
                random.seed(42)
                
                all_customers = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate defaulters
                status_text.text("Generating 750 defaulters...")
                for i in range(1, DEFAULTERS + 1):
                    customer = generate_customer_record(i, is_defaulter=True)
                    all_customers.append(customer)
                    if i % 50 == 0:
                        progress_bar.progress(i / 2500)
                
                # Generate non-defaulters
                status_text.text("Generating 1,750 non-defaulters...")
                for i in range(DEFAULTERS + 1, TOTAL_RECORDS + 1):
                    customer = generate_customer_record(i, is_defaulter=False)
                    all_customers.append(customer)
                    if i % 50 == 0:
                        progress_bar.progress(i / 2500)
                
                # Shuffle
                status_text.text("Shuffling records...")
                random.shuffle(all_customers)
                
                # Reassign IDs
                for idx, cust in enumerate(all_customers, 1):
                    cust['Customer_ID'] = f'CUST-{idx:06d}'
                    cust['Loan_ID'] = f"LOAN-{idx:06d}-01"
                
                # Create DataFrame
                status_text.text("Creating DataFrame...")
                df = pd.DataFrame(all_customers)
                
                # Reorder columns
                column_order = [
                    'Customer_ID', 'Full_Name', 'ID_Number', 'Date_of_Birth', 'Age', 'Gender',
                    'Phone_Number', 'Email', 'Address', 'Location',
                    'Employment_Status', 'Monthly_Income_Bracket', 'Estimated_Monthly_Income',
                    'Account_Opening_Date',
                    'Has_Savings_Account', 'Account_Balance',
                    'Loan_ID', 'Loan_Product', 'Loan_Amount', 'Interest_Rate', 'Term_Type', 
                    'Term_Months', 'Term_Weeks', 'Total_Loan_Amount',
                    'Loan_Disbursement_Date', 'Next_Due_Date', 'Loan_Status',
                    'Outstanding_Balance', 'Total_Paid', 'Overdue_Amount',
                    'Payments_Made', 'Payments_Missed', 'Number_of_Loans',
                    'Number_of_Declined_Loans', 'Declined_Loans_Details',
                    'Profile_Type', 'Is_Defaulter'
                ]
                df = df[column_order]
                
                # Save file
                status_text.text("Saving to file...")
                output_path = Path("data/FKC_Credit_System_Data.csv")
                output_path.parent.mkdir(exist_ok=True)
                df.to_csv(output_path, index=False)
                
                progress_bar.progress(100)
                status_text.empty()
                
                st.success(f"✅ Successfully generated {len(df):,} customer records!")
                
                # Display statistics
                st.markdown("### 📊 Generation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    defaulters_count = len(df[df['Is_Defaulter'] == 'Yes'])
                    st.metric("Defaulters", f"{defaulters_count:,}", 
                             f"{defaulters_count/len(df)*100:.1f}%")
                
                with col2:
                    non_defaulters_count = len(df[df['Is_Defaulter'] == 'No'])
                    st.metric("Non-Defaulters", f"{non_defaulters_count:,}", 
                             f"{non_defaulters_count/len(df)*100:.1f}%")
                
                with col3:
                    avg_loan = df['Loan_Amount'].mean()
                    st.metric("Avg Loan Amount", f"KES {avg_loan:,.0f}")
                
                with col4:
                    avg_age = df['Age'].mean()
                    st.metric("Avg Customer Age", f"{avg_age:.1f} years")
                
                # Store in session state
                st.session_state['data_generated'] = True
                st.session_state['df'] = df
                
            except Exception as e:
                st.error(f"❌ Error during data generation: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

# Display existing data if available
data_path = Path("data/FKC_Credit_System_Data.csv")

if data_path.exists():
    st.markdown("### 📁 Existing Data")
    
    df = pd.read_csv(data_path)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        defaulters = len(df[df['Is_Defaulter'] == 'Yes'])
        st.metric("Defaulters", f"{defaulters:,}", f"{defaulters/len(df)*100:.1f}%")
    
    with col3:
        st.metric("Features", f"{len(df.columns)}")
    
    with col4:
        file_size = data_path.stat().st_size / 1024 / 1024
        st.metric("File Size", f"{file_size:.2f} MB")
    
    # Sample data preview
    st.markdown("#### 👀 Sample Records (First 10)")
    
    display_cols = ['Customer_ID', 'Full_Name', 'Age', 'Gender', 'Location', 
                   'Employment_Status', 'Loan_Product', 'Loan_Amount', 
                   'Loan_Status', 'Is_Defaulter']
    
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    
    # Distribution charts
    with st.expander("📊 View Data Distributions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Location Distribution**")
            location_dist = df['Location'].value_counts()
            st.bar_chart(location_dist)
            
            st.markdown("**Loan Product Distribution**")
            product_dist = df['Loan_Product'].value_counts()
            st.bar_chart(product_dist)
        
        with col2:
            st.markdown("**Employment Status Distribution**")
            employment_dist = df['Employment_Status'].value_counts()
            st.bar_chart(employment_dist)
            
            st.markdown("**Loan Status Distribution**")
            status_dist = df['Loan_Status'].value_counts()
            st.bar_chart(status_dist)
    
    # Download button
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Data",
        data=csv,
        file_name="FKC_Credit_System_Data.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("ℹ️ No data file found. Generate data using the button above.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Data Generation Module | FKC Credit Default Prediction System</p>
    <p style="margin-top: 10px; font-size: 0.9rem;">
        Developed by: Michael Kamau Kibugu (093371) | Strathmore University, Nairobi, Kenya
    </p>
</div>
""", unsafe_allow_html=True)
