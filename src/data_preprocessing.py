import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing for fraud detection."""
    
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.fraud_df = None
        self.ip_country_df = None
        self.credit_df = None
        
    def load_data(self):
        """Load all datasets."""
        print("Loading datasets...")
        
        # Load e-commerce fraud data
        self.fraud_df = pd.read_csv(f'{self.data_path}Fraud_Data.csv')
        print(f"Fraud_Data shape: {self.fraud_df.shape}")
        
        # Load IP to country mapping
        self.ip_country_df = pd.read_csv(f'{self.data_path}IpAddress_to_Country.csv')
        print(f"IpAddress_to_Country shape: {self.ip_country_df.shape}")
        
        # Load credit card data
        self.credit_df = pd.read_csv(f'{self.data_path}creditcard.csv')
        print(f"creditcard shape: {self.credit_df.shape}")
        
        return self.fraud_df, self.ip_country_df, self.credit_df
    
    def clean_fraud_data(self):
        """Clean the e-commerce fraud dataset."""
        print("\nCleaning Fraud_Data...")
        
        # Create a copy
        df = self.fraud_df.copy()
        
        # 1. Check for missing values
        print("Missing values:")
        print(df.isnull().sum())
        
        # 2. Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Removed {duplicates} duplicates")
        
        # 3. Correct data types
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['age'] = df['age'].astype('int')
        
        # 4. Fix target variable typo (o should be 0)
        df['class'] = df['class'].replace('o', 0).astype('int')
        
        # 5. Clean categorical columns
        df['source'] = df['source'].str.strip().str.upper()
        df['browser'] = df['browser'].str.strip()
        df['sex'] = df['sex'].str.upper()
        
        self.fraud_df = df
        print(f"Cleaned shape: {df.shape}")
        return df
    
    def clean_credit_data(self):
        """Clean the credit card dataset."""
        print("\nCleaning creditcard data...")
        
        df = self.credit_df.copy()
        
        # Check for missing values
        print("Missing values:")
        print(df.isnull().sum())
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Ensure Class is integer
        df['Class'] = df['Class'].astype('int')
        
        self.credit_df = df
        print(f"Cleaned shape: {df.shape}")
        return df