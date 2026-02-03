# FKC Credit Default Prediction - Deployment Guide

## 🚀 Quick Start (Local)

### Option 1: Using the Batch File (Windows)
Simply double-click `start_app.bat` in the project folder.

### Option 2: Using Command Line
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud (FREE)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Prepare Your Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "FKC Credit Default Prediction System"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories
4. Click **"New app"**
5. Fill in the details:
   - **Repository**: Select your repository
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. Click **"Deploy!"**

#### 3. Wait for Deployment

- Streamlit will install dependencies from `requirements.txt`
- This takes 2-5 minutes
- You'll see a build log

#### 4. Share Your App

Once deployed, you'll get a URL like:
```
https://YOUR_USERNAME-YOUR_REPO_NAME-app-RANDOM.streamlit.app
```

Share this link with your client/professor!

---

## 📱 Alternative Deployment Options

### Heroku (Free Tier Available)

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Railway (Free Tier Available)

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repo
3. Railway auto-detects Streamlit
4. Deploy!

---

## 🎓 For School Project Presentation

### Demo Workflow

1. **Start with Home Page**
   - Explain the problem (credit default prediction)
   - Show system overview

2. **Data Generation Page**
   - Click "Generate Data"
   - Explain the 2,500 records
   - Show sample data

3. **EDA Analysis Page**
   - Show interactive charts
   - Highlight key insights
   - Demonstrate correlation analysis

4. **Model Training Page**
   - Train all 3 models
   - Show performance comparison
   - Explain metrics (ROC-AUC, Accuracy, etc.)

5. **Prediction Page** ⭐ **MOST IMPORTANT**
   - Enter sample customer data
   - Show real-time prediction
   - Explain risk assessment
   - Demonstrate recommendation system

### Sample Customer for Demo

**Low Risk Customer:**
- Age: 35
- Location: Nairobi
- Employment: Formally employed (permanent)
- Income: 50,001 - 100,000
- Savings: Yes
- Loan Amount: 50,000
- Payments Made: 10
- Payments Missed: 0
- Overdue: 0

**High Risk Customer:**
- Age: 28
- Location: Kayole
- Employment: Casual laborer
- Income: Below 10,000
- Savings: No
- Loan Amount: 80,000
- Payments Made: 2
- Payments Missed: 5
- Overdue: 15,000

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Module Not Found
```bash
pip install -r requirements.txt
```

### Model Not Found Error
1. Go to "Model Training" page
2. Train at least one model
3. Return to "Make Predictions" page

### Streamlit Not Opening
- Check if port 8501 is blocked
- Try different port
- Check firewall settings

---

## 📊 Performance Tips

### For Large Datasets
- Use `@st.cache_data` decorator (already implemented)
- Limit data preview to first 1000 rows
- Use sampling for visualizations

### For Faster Training
- Reduce GridSearchCV parameter grid
- Use fewer cross-validation folds
- Train only one model at a time

---

## 🔒 Security Notes

### For Production Deployment

1. **Environment Variables**
   - Don't hardcode sensitive data
   - Use Streamlit secrets for API keys

2. **Data Privacy**
   - Customer data is already masked
   - Don't expose real customer information

3. **Access Control**
   - Use Streamlit authentication (paid feature)
   - Or deploy behind a VPN/firewall

---

## 📞 Support

If you encounter issues during deployment:

1. Check Streamlit Community Forum
2. Review deployment logs
3. Ensure all dependencies are in requirements.txt
4. Test locally first before deploying

---

## ✅ Pre-Deployment Checklist

- [ ] All dependencies in requirements.txt
- [ ] Code pushed to GitHub
- [ ] README.md updated
- [ ] .gitignore configured
- [ ] App tested locally
- [ ] Sample data generated
- [ ] Model trained
- [ ] Predictions working

---

**Good luck with your presentation! 🎓**

The Streamlit app makes your school project look professional and impressive!
