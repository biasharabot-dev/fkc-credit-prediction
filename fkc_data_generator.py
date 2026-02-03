"""
Frontier Kenya Credit Limited (FKC) - Credit System Data Generator
Generates 2,500 realistic customer records from credit system database extract
Combines static customer data, financial data, and loan data
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with locale (using en_US as en_KE is not available)
fake = Faker('en_US')
np.random.seed(42)
random.seed(42)

# Configuration
TOTAL_RECORDS = 2500
NON_DEFAULTERS = 1750  # 70%
DEFAULTERS = 750  # 30%

# FKC Operating Locations (not all counties)
OPERATING_LOCATIONS = {
    'Nairobi': 0.50,  # 50% - main location
    'Kayole': 0.20,   # 20%
    'Kamulu': 0.15,   # 15%
    'Mombasa Road': 0.10,  # 10%
    'Mwea': 0.05      # 5%
}

# Loan Products with fixed interest rates
LOAN_PRODUCTS = {
    'School Fees Loan': {
        'interest_rate': 14.5,
        'typical_amount_range': (20000, 150000),
        'typical_term_months': [3, 6, 9],
        'typical_term_weeks': [12, 24, 36]
    },
    'Utility Loan': {
        'interest_rate': 16.0,
        'typical_amount_range': (10000, 50000),
        'typical_term_months': [3, 6],
        'typical_term_weeks': [12, 24]
    },
    'Cash Loan': {
        'interest_rate': 18.5,
        'typical_amount_range': (15000, 200000),
        'typical_term_months': [3, 6, 9, 12],
        'typical_term_weeks': [12, 24, 36, 48]
    },
    'Business Capital Loan': {
        'interest_rate': 19.5,
        'typical_amount_range': (50000, 300000),
        'typical_term_months': [6, 9, 12],
        'typical_term_weeks': [24, 36, 48]
    },
    'Emergency Loan': {
        'interest_rate': 22.0,
        'typical_amount_range': (10000, 100000),
        'typical_term_months': [3, 6],
        'typical_term_weeks': [12, 24]
    }
}

# Loan Decline Reasons
DECLINE_REASONS = [
    'Insufficient income',
    'Poor credit history',
    'Too many existing loans',
    'Age limit exceeded',
    'Incomplete documentation',
    'High debt-to-income ratio',
    'Unstable employment',
    'Loan amount too high'
]

# Borrower Profile Definitions (for generating realistic data)
PROFILES = {
    'Stable Professional': {
        'weight': 0.25,
        'default_rate': 0.15,
        'age_range': (30, 45),
        'employment_dist': {'Formally employed (permanent)': 0.85, 'Formally employed (contract)': 0.15},
        'income_dist': {'50,001 - 100,000': 0.5, 'Above 100,000': 0.35, '30,001 - 50,000': 0.15},
        'loan_amount_range': (50000, 500000),
        'savings_probability': 0.9
    },
    'Young Entrepreneur': {
        'weight': 0.20,
        'default_rate': 0.25,
        'age_range': (25, 35),
        'employment_dist': {'Self-employed/Business owner': 0.7, 'Formally employed (contract)': 0.3},
        'income_dist': {'20,001 - 30,000': 0.3, '30,001 - 50,000': 0.5, '50,001 - 100,000': 0.2},
        'loan_amount_range': (30000, 200000),
        'savings_probability': 0.6
    },
    'Hustler/Casual Worker': {
        'weight': 0.25,
        'default_rate': 0.45,
        'age_range': (22, 38),
        'employment_dist': {'Casual laborer': 0.4, 'Self-employed/Business owner': 0.4, 'Formally employed (contract)': 0.2},
        'income_dist': {'Below 10,000': 0.2, '10,000 - 20,000': 0.45, '20,001 - 30,000': 0.35},
        'loan_amount_range': (10000, 100000),
        'savings_probability': 0.25
    },
    'Overburdened Family Person': {
        'weight': 0.20,
        'default_rate': 0.40,
        'age_range': (35, 50),
        'employment_dist': {'Formally employed (contract)': 0.35, 'Self-employed/Business owner': 0.4, 'Formally employed (permanent)': 0.25},
        'income_dist': {'20,001 - 30,000': 0.4, '30,001 - 50,000': 0.45, '50,001 - 100,000': 0.15},
        'loan_amount_range': (20000, 150000),
        'savings_probability': 0.4
    },
    'Entry Level Worker': {
        'weight': 0.10,
        'default_rate': 0.20,
        'age_range': (21, 30),
        'employment_dist': {'Formally employed (contract)': 0.7, 'Formally employed (permanent)': 0.3},
        'income_dist': {'10,000 - 20,000': 0.45, '20,001 - 30,000': 0.4, '30,001 - 50,000': 0.15},
        'loan_amount_range': (15000, 80000),
        'savings_probability': 0.5
    }
}

def weighted_choice(choices_dict):
    """Select item based on weighted probabilities"""
    items = list(choices_dict.keys())
    weights = list(choices_dict.values())
    return random.choices(items, weights=weights, k=1)[0]

def assign_location():
    """Assign location based on FKC operating areas"""
    locations = list(OPERATING_LOCATIONS.keys())
    weights = list(OPERATING_LOCATIONS.values())
    return random.choices(locations, weights=weights, k=1)[0]

def generate_kenyan_phone():
    """Generate a Kenyan phone number in format +254 7XX XXX XXX"""
    # Kenyan mobile prefixes (7XX)
    prefix = random.choice(['70', '71', '72', '73', '74', '75', '76', '77', '78', '79'])
    # Generate 7 more digits
    suffix = ''.join([str(random.randint(0, 9)) for _ in range(7)])
    return f"+254 {prefix}{suffix[0]} {suffix[1:4]} {suffix[4:]}"

def mask_id_number(id_number):
    """Mask ID number showing only first 3 and last 2 characters"""
    if len(id_number) >= 5:
        return f"{id_number[:3]}***{id_number[-2:]}"
    return "***" + id_number[-2:] if len(id_number) >= 2 else "***"

def mask_phone_number(phone_number):
    """Mask phone number showing only country code and last 3 digits"""
    # Extract last 3 digits
    digits_only = ''.join(filter(str.isdigit, phone_number))
    if len(digits_only) >= 3:
        return f"+254 *** *** {digits_only[-3:]}"
    return "+254 *** *** ***"

# Common email domains
EMAIL_DOMAINS = [
    'gmail.com',
    'yahoo.com',
    'outlook.com',
    'hotmail.com',
    'yahoo.co.uk',
    'icloud.com',
    'aol.com',
    'protonmail.com',
    'mail.com',
    'yandex.com',
    'gmail.co.ke',
    'yahoo.co.ke',
    'outlook.co.ke'
]

def mask_email(email):
    """Mask email address and use real email domains"""
    if not email:
        return None
    # Extract username part (before @)
    if '@' in email:
        username = email.split('@')[0]
        # Mask username showing first 2 chars and last 1 char
        if len(username) > 3:
            masked_username = f"{username[:2]}***{username[-1]}"
        else:
            masked_username = "***"
        # Use real email domain (randomly selected)
        domain = random.choice(EMAIL_DOMAINS)
        return f"{masked_username}@{domain}"
    return f"***@{random.choice(EMAIL_DOMAINS)}"

def get_loan_status(is_defaulter, loan_disbursement_date):
    """Determine loan status based on defaulter status and time"""
    days_since_disbursement = (datetime.now() - loan_disbursement_date).days
    
    if is_defaulter:
        if days_since_disbursement < 30:
            return np.random.choice(['Active', 'Overdue'], p=[0.7, 0.3])
        elif days_since_disbursement < 90:
            return np.random.choice(['Overdue', 'Defaulted'], p=[0.4, 0.6])
        else:
            return np.random.choice(['Defaulted', 'Written Off'], p=[0.7, 0.3])
    else:
        if days_since_disbursement < 30:
            return 'Active'
        elif days_since_disbursement < 90:
            return np.random.choice(['Active', 'Overdue'], p=[0.9, 0.1])
        else:
            return np.random.choice(['Active', 'Completed'], p=[0.3, 0.7])

def calculate_repayment_history(loan_amount, interest_rate, term_months, term_weeks, is_defaulter, disbursement_date, term_type='monthly'):
    """Calculate repayment history for monthly or weekly terms"""
    if term_type == 'weekly':
        total_payments = term_weeks
        payment_period_days = 7
        payment_amount = (loan_amount * (1 + interest_rate/100)) / term_weeks
        periods_elapsed = min((datetime.now() - disbursement_date).days // 7, term_weeks)
    else:  # monthly
        total_payments = term_months
        payment_period_days = 30
        payment_amount = (loan_amount * (1 + interest_rate/100)) / term_months
        periods_elapsed = min((datetime.now() - disbursement_date).days // 30, term_months)
    
    if is_defaulter:
        # Defaulters miss some payments
        payments_made = max(0, int(periods_elapsed * random.uniform(0.3, 0.7)))
        total_paid = payments_made * payment_amount * random.uniform(0.8, 1.0)  # Sometimes partial payments
        overdue_amount = max(0, (periods_elapsed - payments_made) * payment_amount)
    else:
        # Non-defaulters make most payments
        payments_made = int(periods_elapsed * random.uniform(0.85, 1.0))
        total_paid = payments_made * payment_amount
        overdue_amount = max(0, (periods_elapsed - payments_made) * payment_amount * random.uniform(0, 0.2))
    
    return {
        'total_paid': round(total_paid, 2),
        'overdue_amount': round(overdue_amount, 2),
        'payments_made': payments_made,
        'payments_missed': max(0, periods_elapsed - payments_made),
        'payment_period_days': payment_period_days
    }

def generate_customer_record(customer_id, is_defaulter):
    """Generate a single customer record with static, financial, and loan data"""
    
    # Select profile based on weights, adjusted for defaulter status
    if is_defaulter:
        profile_weights = {
            'Stable Professional': 0.15,
            'Young Entrepreneur': 0.20,
            'Hustler/Casual Worker': 0.35,
            'Overburdened Family Person': 0.25,
            'Entry Level Worker': 0.05
        }
    else:
        profile_weights = {
            'Stable Professional': 0.30,
            'Young Entrepreneur': 0.20,
            'Hustler/Casual Worker': 0.18,
            'Overburdened Family Person': 0.17,
            'Entry Level Worker': 0.15
        }
    
    profile_name = weighted_choice(profile_weights)
    profile = PROFILES[profile_name]
    
    # Static Customer Data
    # Include older customers (50-70) with low probability (5%)
    if random.random() < 0.05:
        age = random.randint(50, 70)
        is_older_customer = True
    else:
        age = random.randint(*profile['age_range'])
        is_older_customer = False
    
    date_of_birth = datetime.now() - timedelta(days=age*365 + random.randint(0, 365))
    gender = random.choice(['Male', 'Female'])
    # Generate gender-appropriate name
    if gender == 'Male':
        full_name = fake.name_male()
    else:
        full_name = fake.name_female()
    phone_number_raw = generate_kenyan_phone()
    phone_number = mask_phone_number(phone_number_raw)
    email_raw = fake.email() if random.random() < 0.6 else None  # 40% don't have email
    email = mask_email(email_raw)
    id_number_raw = fake.bothify(text='#########', letters='ABCDEFGHJKLMNPQRSTUVWXYZ')
    id_number = mask_id_number(id_number_raw)
    
    location = assign_location()
    address = f"{fake.street_address()}, {location}"
    
    employment_status = weighted_choice(profile['employment_dist'])
    income_bracket = weighted_choice(profile['income_dist'])
    
    # Income midpoints for calculations
    income_midpoints = {
        'Below 10,000': 7500,
        '10,000 - 20,000': 15000,
        '20,001 - 30,000': 25000,
        '30,001 - 50,000': 40000,
        '50,001 - 100,000': 75000,
        'Above 100,000': 150000
    }
    estimated_monthly_income = income_midpoints.get(income_bracket, 25000)
    
    # Account opening date (customer registered with FKC)
    account_opening_date = datetime.now() - timedelta(days=random.randint(30, 1095))  # 1 month to 3 years ago
    
    # Financial Data
    has_savings_account = random.random() < profile['savings_probability']
    if has_savings_account:
        account_balance = random.uniform(1000, estimated_monthly_income * 3)
    else:
        account_balance = random.uniform(0, 5000)
    
    # Loan Data
    # Generate more loans per customer (more realistic distribution)
    # 50% have 1 loan, 30% have 2, 15% have 3, 5% have 4+
    num_loans = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
    
    loans = []
    declined_loans = []
    
    # Determine if customer has any declined loans (20% of customers have at least one declined loan)
    has_declined_loans = random.random() < 0.20
    num_declined = random.randint(1, 2) if has_declined_loans else 0
    
    # Generate declined loans first (if any)
    for decline_num in range(num_declined):
        # Select loan product
        product_name = random.choice(list(LOAN_PRODUCTS.keys()))
        product = LOAN_PRODUCTS[product_name]
        
        # Loan amount based on product
        loan_amount = random.uniform(*product['typical_amount_range'])
        loan_amount = round(loan_amount / 1000) * 1000
        
        # Decline reason
        decline_reason = random.choice(DECLINE_REASONS)
        
        # Decline date (before account opening or early in account history)
        days_since_account = (datetime.now() - account_opening_date).days
        decline_days_ago = random.randint(0, min(days_since_account, 180))
        decline_date = datetime.now() - timedelta(days=decline_days_ago)
        
        declined_loans.append({
            'loan_product': product_name,
            'loan_amount': loan_amount,
            'interest_rate': product['interest_rate'],
            'decline_date': decline_date.strftime('%Y-%m-%d'),
            'decline_reason': decline_reason
        })
    
    # Generate approved loans
    for loan_num in range(num_loans):
        # Select loan product
        product_name = random.choice(list(LOAN_PRODUCTS.keys()))
        product = LOAN_PRODUCTS[product_name]
        
        # Loan amount based on product (but constrained by profile)
        product_min, product_max = product['typical_amount_range']
        profile_min, profile_max = profile['loan_amount_range']
        loan_amount = random.uniform(max(product_min, profile_min), min(product_max, profile_max))
        loan_amount = round(loan_amount / 1000) * 1000
        
        # Interest rate from product (higher for older customers)
        base_interest_rate = product['interest_rate']
        if is_older_customer:
            # Add fixed 3% premium for older customers (round to 2 decimals)
            interest_rate = round(base_interest_rate + 3.0, 2)
        else:
            interest_rate = base_interest_rate
        
        # Determine term type (70% monthly, 30% weekly)
        term_type = random.choice(['monthly', 'weekly'])
        
        if term_type == 'weekly':
            term_weeks = random.choice(product['typical_term_weeks'])
            term_months = None
        else:
            term_months = random.choice(product['typical_term_months'])
            term_weeks = None
        
        # Disbursement date (spread over customer's account lifetime)
        days_since_account = (datetime.now() - account_opening_date).days
        days_ago = random.randint(0, min(days_since_account, 365))
        loan_disbursement_date = datetime.now() - timedelta(days=days_ago)
        
        # Loan status
        loan_status = get_loan_status(is_defaulter, loan_disbursement_date)
        
        # Calculate repayment details
        repayment_info = calculate_repayment_history(
            loan_amount, interest_rate, term_months, term_weeks, is_defaulter, 
            loan_disbursement_date, term_type
        )
        
        # Due date (next payment due)
        if loan_status in ['Active', 'Overdue']:
            payment_period_days = repayment_info['payment_period_days']
            last_payment_count = repayment_info['payments_made']
            next_due_date = loan_disbursement_date + timedelta(days=(last_payment_count + 1) * payment_period_days)
        else:
            next_due_date = None
        
        # Total loan amount with interest
        total_loan_amount = loan_amount * (1 + interest_rate/100)
        outstanding_balance = total_loan_amount - repayment_info['total_paid']
        
        loans.append({
            'loan_id': f'LOAN-{customer_id:06d}-{loan_num+1:02d}',
            'loan_product': product_name,
            'loan_amount': loan_amount,
            'interest_rate': round(interest_rate, 2),
            'term_type': term_type,
            'term_months': term_months,
            'term_weeks': term_weeks,
            'total_loan_amount': round(total_loan_amount, 2),
            'disbursement_date': loan_disbursement_date.strftime('%Y-%m-%d'),
            'next_due_date': next_due_date.strftime('%Y-%m-%d') if next_due_date else None,
            'loan_status': loan_status,
            'outstanding_balance': round(max(0, outstanding_balance), 2),
            'total_paid': repayment_info['total_paid'],
            'overdue_amount': repayment_info['overdue_amount'],
            'payments_made': repayment_info['payments_made'],
            'payments_missed': repayment_info['payments_missed']
        })
    
    # Format declined loans information
    if declined_loans:
        declined_info = '; '.join([
            f"{d['loan_product']} ({d['loan_amount']:,.0f} KES) - {d['decline_reason']}"
            for d in declined_loans
        ])
        num_declined_loans = len(declined_loans)
    else:
        declined_info = None
        num_declined_loans = 0
    
    # Create customer record
    customer_record = {
        # Static Data
        'Customer_ID': f'CUST-{customer_id:06d}',
        'Full_Name': full_name,
        'ID_Number': id_number,
        'Date_of_Birth': date_of_birth.strftime('%Y-%m-%d'),
        'Age': age,
        'Gender': gender,
        'Phone_Number': phone_number,
        'Email': email,
        'Address': address,
        'Location': location,
        'Employment_Status': employment_status,
        'Monthly_Income_Bracket': income_bracket,
        'Estimated_Monthly_Income': round(estimated_monthly_income, 2),
        'Account_Opening_Date': account_opening_date.strftime('%Y-%m-%d'),
        
        # Financial Data
        'Has_Savings_Account': 'Yes' if has_savings_account else 'No',
        'Account_Balance': round(account_balance, 2),
        
        # Loan Data (primary/most recent loan)
        'Loan_ID': loans[0]['loan_id'],
        'Loan_Product': loans[0]['loan_product'],
        'Loan_Amount': loans[0]['loan_amount'],
        'Interest_Rate': loans[0]['interest_rate'],
        'Term_Type': loans[0]['term_type'],
        'Term_Months': loans[0]['term_months'],
        'Term_Weeks': loans[0]['term_weeks'],
        'Total_Loan_Amount': loans[0]['total_loan_amount'],
        'Loan_Disbursement_Date': loans[0]['disbursement_date'],
        'Next_Due_Date': loans[0]['next_due_date'],
        'Loan_Status': loans[0]['loan_status'],
        'Outstanding_Balance': loans[0]['outstanding_balance'],
        'Total_Paid': loans[0]['total_paid'],
        'Overdue_Amount': loans[0]['overdue_amount'],
        'Payments_Made': loans[0]['payments_made'],
        'Payments_Missed': loans[0]['payments_missed'],
        'Number_of_Loans': num_loans,
        
        # Declined Loans
        'Number_of_Declined_Loans': num_declined_loans,
        'Declined_Loans_Details': declined_info,
        
        # Target Variable
        'Is_Defaulter': 'Yes' if is_defaulter else 'No',
        'Profile_Type': profile_name  # For analysis
    }
    
    return customer_record

# Generate all customer records
print("Generating 2,500 customer records from credit system...")
all_customers = []

# Generate defaulters
print("Generating 750 defaulters...")
for i in range(1, DEFAULTERS + 1):
    customer = generate_customer_record(i, is_defaulter=True)
    all_customers.append(customer)

# Generate non-defaulters
print("Generating 1,750 non-defaulters...")
for i in range(DEFAULTERS + 1, TOTAL_RECORDS + 1):
    customer = generate_customer_record(i, is_defaulter=False)
    all_customers.append(customer)

# Shuffle to mix defaulters and non-defaulters
random.shuffle(all_customers)

# Reassign Customer IDs sequentially after shuffle
for idx, cust in enumerate(all_customers, 1):
    cust['Customer_ID'] = f'CUST-{idx:06d}'
    # Update loan IDs to match new customer ID
    old_loan_id = cust['Loan_ID']
    cust['Loan_ID'] = f"LOAN-{idx:06d}-01"

# Create DataFrame
df = pd.DataFrame(all_customers)

# Reorder columns for better presentation
column_order = [
    # Static Data
    'Customer_ID', 'Full_Name', 'ID_Number', 'Date_of_Birth', 'Age', 'Gender',
    'Phone_Number', 'Email', 'Address', 'Location',
    'Employment_Status', 'Monthly_Income_Bracket', 'Estimated_Monthly_Income',
    'Account_Opening_Date',
    # Financial Data
    'Has_Savings_Account', 'Account_Balance',
    # Loan Data
    'Loan_ID', 'Loan_Product', 'Loan_Amount', 'Interest_Rate', 'Term_Type', 
    'Term_Months', 'Term_Weeks', 'Total_Loan_Amount',
    'Loan_Disbursement_Date', 'Next_Due_Date', 'Loan_Status',
    'Outstanding_Balance', 'Total_Paid', 'Overdue_Amount',
    'Payments_Made', 'Payments_Missed', 'Number_of_Loans',
    # Declined Loans
    'Number_of_Declined_Loans', 'Declined_Loans_Details',
    # Analysis Fields
    'Profile_Type', 'Is_Defaulter'
]

df = df[column_order]

# Save file
print("\nSaving file...")
df.to_csv('FKC_Credit_System_Data.csv', index=False)

print("\n" + "="*60)
print("DATA GENERATION COMPLETE!")
print("="*60)
print(f"\nTotal records generated: {len(df)}")
print(f"Defaulters: {df['Is_Defaulter'].value_counts().get('Yes', 0)} ({df['Is_Defaulter'].value_counts().get('Yes', 0)/len(df)*100:.1f}%)")
print(f"Non-defaulters: {df['Is_Defaulter'].value_counts().get('No', 0)} ({df['Is_Defaulter'].value_counts().get('No', 0)/len(df)*100:.1f}%)")

print("\nLocation Distribution:")
print(df['Location'].value_counts())

print("\nLoan Status Distribution:")
print(df['Loan_Status'].value_counts())

print("\nProfile Distribution:")
print(df['Profile_Type'].value_counts())

print("\nAge Distribution:")
print(f"Customers 50+: {len(df[df['Age'] >= 50])} ({len(df[df['Age'] >= 50])/len(df)*100:.1f}%)")
print(f"Age range: {df['Age'].min()} - {df['Age'].max()} years")

print("\nLoan Product Distribution:")
print(df['Loan_Product'].value_counts())

print("\nTerm Type Distribution:")
print(df['Term_Type'].value_counts())

print("\nDeclined Loans:")
print(f"Customers with declined loans: {len(df[df['Number_of_Declined_Loans'] > 0])} ({len(df[df['Number_of_Declined_Loans'] > 0])/len(df)*100:.1f}%)")
print(f"Total declined loans: {df['Number_of_Declined_Loans'].sum()}")

print("\nInterest Rate Distribution:")
print(df['Interest_Rate'].value_counts().sort_index())

print("\nNumber of Loans Distribution:")
print(df['Number_of_Loans'].value_counts().sort_index())

print("\nFile saved:")
print("  - FKC_Credit_System_Data.csv (credit system database extract)")

print("\nSample records (first 5):")
print(df[['Customer_ID', 'Full_Name', 'Age', 'Location', 'Loan_Product', 'Loan_Amount', 'Term_Type', 'Interest_Rate', 'Loan_Status', 'Is_Defaulter']].head())

print("\n" + "="*60)
