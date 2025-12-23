# Fraud Detection ML Pipeline 🚨

A comprehensive machine learning pipeline for detecting fraudulent transactions in e-commerce and banking data. This project implements feature engineering, handles severe class imbalance, and uses SHAP for model explainability to balance fraud prevention with user experience.

## 📋 Project Overview

**Business Context:** You are a data scientist at **Adey Innovations Inc.**, tasked with improving fraud detection for e-commerce transactions and bank credit transactions. The key challenge is managing the trade-off between security and user experience—minimizing false positives (legitimate transactions flagged as fraud) while catching actual fraudulent activities.

**Datasets:**
1. **Fraud_Data.csv** - E-commerce transactions with user behavior data
2. **IpAddress_to_Country.csv** - IP address to country mapping
3. **creditcard.csv** - Bank transaction data with PCA-transformed features

## 🎯 Project Goals

- Build accurate fraud detection models for both e-commerce and banking transactions
- Handle extreme class imbalance
- Implement geolocation analysis using IP addresses
- Create actionable business recommendations through model explainability
- Balance detection accuracy with user experience considerations

## 📁 Project Structure
```
fraud-detection-ml-pipeline/
├── data/                          # ⚠️ Add to .gitignore
│   ├── raw/                       # Original datasets
│   └── processed/                 # Cleaned and engineered data
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_eda-fraud-data.ipynb
│   ├── 02_eda-creditcard.ipynb
│   └── 03_feature-engineering.ipynb
├── src/                           # Reusable Python modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── ip_utils.py
│   └── utils.py
├── scripts/                       # Standalone executable scripts
│   └── run_preprocessing.py
├── models/                        # Saved model artifacts
├── reports/                       # Generated reports and figures
│   └── figures/
├── requirements.txt               # Python dependencies
├── README.md                      
└── .gitignore
```


## 🚀 Quick Start

### 1. Prerequisites

```bash
# Clone the repository
git clone https://github.com/Jaki77/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
