# FKC Credit Default Prediction System 💳

An intelligent **Machine Learning-powered** credit risk assessment platform for **Frontier Kenya Credit Limited (FKC)**. This system predicts credit default probability using advanced analytics and provides real-time risk assessment for lending decisions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **📊 Data Generation**: Generate 2,500 realistic customer credit records with demographics, financial history, and loan details.
- **🔍 EDA Analysis**: Comprehensive exploratory data analysis with interactive visualizations.
- **🤖 ML Models**: Train and compare 3 machine learning models (Logistic Regression, Random Forest, SVM).
- **🎯 Predictions**: Real-time credit default risk prediction with probability scores
- **📈 Visualizations**: Interactive charts and graphs using Plotly
- **💾 Model Persistence**: Save and load trained models for production use

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

```bash
cd system_dev_final
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app.py
```

4. **Open your browser**

The app will automatically open at `http://localhost:8501`

## 📖 User Guide

### Step 1: Generate Data 📊

1. Navigate to **Data Generation** page
2. Click **"Generate Customer Data"** button
3. Wait for 2,500 records to be generated
4. View sample data and distributions

### Step 2: Explore Data 🔍

1. Go to **EDA Analysis** page
2. Explore different tabs:
   - Target Analysis
   - Feature Distributions
   - Correlations
   - Default Analysis
   - Data Quality
3. Download processed data if needed

### Step 3: Train Models 🤖

1. Navigate to **Model Training** page
2. Configure training parameters:
   - Test set size (default: 20%)
   - Cross-validation folds (default: 5)
   - Random state (default: 42)
3. Select models to train
4. Click **"Start Training"**
5. View performance comparison and metrics
6. Download trained models

### Step 4: Make Predictions 🎯

1. Go to **Make Predictions** page
2. Enter customer information:
   - Personal details (age, gender, location, employment)
   - Financial information (income, account balance, loans)
   - Loan details (product, amount, term, status)
   - Payment history (payments made/missed, overdue amount)
3. Click **"Predict Default Risk"**
4. View results:
   - Default probability
   - Risk level (Low/Medium/High)
   - Recommendation (Approve/Review/Reject)
   - Key risk factors
   - Customer summary

## 🏗️ Project Structure

```
system_dev_final/
├── app.py                          # Main Streamlit application
├── fkc_data_generator.py          # Data generation script
├── eda_and_preprocessing.py       # EDA analysis script
├── model_development.py           # Model training script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── pages/                         # Streamlit pages
│   ├── 1_📊_Data_Generation.py
│   ├── 2_🔍_EDA_Analysis.py
│   ├── 3_🤖_Model_Training.py
│   └── 4_🎯_Make_Predictions.py
│
├── data/                          # Generated data
│   └── FKC_Credit_System_Data.csv
│
├── models/                        # Trained models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
│
└── outputs/                       # Visualizations & results
    └── EDA_Visualizations.png
```

## 🎓 System Overview

### FKC Operating Locations

- 🏙️ **Nairobi** (Main Branch) - 50%
- 🏘️ **Kayole** - 20%
- 🏘️ **Kamulu** - 15%
- 🛣️ **Mombasa Road** - 10%
- 🌾 **Mwea** - 5%

### Loan Products

| Product | Interest Rate |
|---------|--------------|
| School Fees Loan | 14.5% |
| Utility Loan | 16.0% |
| Cash Loan | 18.5% |
| Business Capital Loan | 19.5% |
| Emergency Loan | 22.0% |

### Customer Profiles

| Profile | Default Rate | Distribution |
|---------|-------------|--------------|
| Stable Professional | 15% | 25% |
| Young Entrepreneur | 25% | 20% |
| Hustler/Casual Worker | 45% | 25% |
| Overburdened Family Person | 40% | 20% |
| Entry Level Worker | 20% | 10% |

## 🤖 Machine Learning Models

The system trains and compares three models:

1. **Logistic Regression**
   - Fast training
   - Interpretable coefficients
   - Good baseline performance

2. **Random Forest**
   - High accuracy
   - Feature importance analysis
   - Handles non-linear relationships

3. **Support Vector Machine (SVM)**
   - Robust to outliers
   - Effective in high-dimensional space
   - Multiple kernel options

### Model Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Precision**: Accuracy of default predictions
- **Recall**: Ability to identify all defaulters
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall model performance (primary metric)

## 📊 Features Used

### Numerical Features (18)
- Age, Income, Account Balance
- Loan Amount, Interest Rate
- Outstanding Balance, Total Paid, Overdue Amount
- Payments Made/Missed
- Number of Loans, Declined Loans
- Engineered: Debt-to-Income Ratio, Payment Rate, Loan-to-Income Ratio, etc.

### Categorical Features (8)
- Gender, Location
- Employment Status, Income Bracket
- Savings Account, Loan Product
- Term Type, Loan Status

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Share the link**
   - Your app will be available at: `https://your-app.streamlit.app`

## 🎯 Use Cases

- **Loan Approval**: Automated risk assessment for new loan applications
- **Portfolio Management**: Identify high-risk existing customers
- **Credit Scoring**: Generate credit scores based on ML predictions
- **Risk Analytics**: Analyze default patterns across customer segments
- **Decision Support**: Provide data-driven recommendations to loan officers

## 📝 Notes for School Project

### Presentation Tips

1. **Demo Flow**:
   - Start with home page overview
   - Show data generation process
   - Highlight EDA insights
   - Demonstrate model training
   - **Focus on prediction page** (most impressive)

2. **Key Points to Emphasize**:
   - Real-world application (FKC is a real microfinance institution)
   - Complete ML pipeline (data → EDA → training → prediction)
   - Interactive web interface (not just code)
   - Professional visualizations
   - Practical business value

3. **Technical Highlights**:
   - Multiple ML algorithms compared
   - Hyperparameter tuning with GridSearchCV
   - Feature engineering
   - Cross-validation
   - Model persistence

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

**Issue**: Port already in use
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**Issue**: Model not found
```bash
# Solution: Train model first from Model Training page
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

**Michael Kamau Kibugu**  
Student ID: 093371  
Strathmore University  
Nairobi, Kenya

**Project Type**: School Project - FKC Credit Default Prediction System  
**Year**: 2026

## 🙏 Acknowledgments

- Frontier Kenya Credit Limited for the use case
- Streamlit for the amazing framework
- scikit-learn for ML tools
- Plotly for interactive visualizations

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Contact the development team

---

**Built with ❤️ using Streamlit & Machine Learning**

© 2026 FKC Credit Default Prediction System
