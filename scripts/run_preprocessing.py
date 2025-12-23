"""
Main script to execute Task 1: Data Analysis and Preprocessing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.ip_utils import merge_with_ip_country, validate_ip_addresses, ip_to_int
from src.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("TASK 1: DATA ANALYSIS AND PREPROCESSING")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports/figures/eda_plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and clean data
    preprocessor = DataPreprocessor()
    fraud_df, ip_country_df, credit_df = preprocessor.load_data()
    
    fraud_clean = preprocessor.clean_fraud_data()
    credit_clean = preprocessor.clean_credit_data()
    
    # Step 1.5: Validate IP addresses before merging
    print("\n" + "-"*40)
    print("IP ADDRESS VALIDATION")
    print("-"*40)
    
    valid_ips, invalid_ips = validate_ip_addresses(fraud_clean)
    
    # Test IP conversion on sample
    print("\nTesting IP conversion on sample IPs:")
    sample_ips = fraud_clean['ip_address'].head(5).tolist()
    for ip in sample_ips:
        ip_int = ip_to_int(ip)
        print(f"  {ip} -> {ip_int}")
    
    # Step 2: Geolocation integration
    print("\n" + "-"*40)
    print("GEOLOCATION INTEGRATION")
    print("-"*40)
    
    fraud_with_country = merge_with_ip_country(fraud_clean, ip_country_df)
    
    # If merge failed or returned empty, use fallback
    if fraud_with_country is None or len(fraud_with_country) == 0:
        print("\n⚠️  Merge failed. Using fallback strategy...")
        fraud_with_country = fraud_clean.copy()
        fraud_with_country['country'] = 'Unknown'
    
    # Step 3: Feature engineering
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineer()
    
    # Create all features
    fraud_engineered = fe.create_time_features(fraud_with_country)
    fraud_engineered = fe.create_transaction_velocity(fraud_engineered)
    fraud_engineered = fe.create_purchase_patterns(fraud_engineered)
    fraud_engineered = fe.create_device_features(fraud_engineered)
    
    print(f"\n✅ Created {len(fe.features_created)} new features")
    
    # Step 4: Save processed data
    fraud_engineered.to_csv('data/processed/fraud_data_engineered.csv', index=False)
    credit_clean.to_csv('data/processed/creditcard_processed.csv', index=False)
    
    # Step 5: Generate summary report
    generate_summary_report(fraud_engineered, credit_clean, len(invalid_ips))
    
    print("\n" + "="*60)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutputs saved to:")
    print("  - data/processed/fraud_data_engineered.csv")
    print("  - data/processed/creditcard_processed.csv")
    print("  - reports/task1_summary.txt")

def generate_summary_report(fraud_df, credit_df, invalid_ip_count=0):
    """Generate a summary report of the preprocessing."""
    
    summary = []
    summary.append("="*60)
    summary.append("TASK 1 SUMMARY REPORT")
    summary.append("="*60)
    
    summary.append(f"\n1. DATASET SIZES:")
    summary.append(f"   Fraud Data: {fraud_df.shape[0]} rows, {fraud_df.shape[1]} columns")
    summary.append(f"   Credit Data: {credit_df.shape[0]} rows, {credit_df.shape[1]} columns")
    
    summary.append(f"\n2. CLASS DISTRIBUTION:")
    fraud_rate = fraud_df['class'].mean() * 100
    credit_fraud_rate = credit_df['Class'].mean() * 100
    summary.append(f"   E-commerce fraud rate: {fraud_rate:.4f}%")
    summary.append(f"   Credit card fraud rate: {credit_fraud_rate:.4f}%")
    
    summary.append(f"\n3. DATA QUALITY:")
    summary.append(f"   Missing values in fraud data: {fraud_df.isnull().sum().sum()}")
    summary.append(f"   Missing values in credit data: {credit_df.isnull().sum().sum()}")
    summary.append(f"   Invalid IP addresses: {invalid_ip_count}")
    
    # Country analysis
    if 'country' in fraud_df.columns:
        country_stats = fraud_df['country'].value_counts()
        summary.append(f"\n4. GEOGRAPHIC DISTRIBUTION:")
        summary.append(f"   Transactions with country data: {fraud_df['country'].notnull().sum()}")
        summary.append(f"   Unique countries: {fraud_df['country'].nunique()}")
        
        if len(country_stats) > 0:
            fraud_by_country = fraud_df.groupby('country')['class'].mean().sort_values(ascending=False)
            top_5 = fraud_by_country.head(5)
            summary.append(f"\n5. TOP 5 COUNTRIES BY FRAUD RATE:")
            for country, rate in top_5.items():
                summary.append(f"   {country}: {rate*100:.2f}%")
    
    summary.append(f"\n6. KEY FEATURES CREATED:")
    # Get engineered features (non-original columns)
    original_cols = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                    'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class']
    engineered_features = [col for col in fraud_df.columns if col not in original_cols]
    
    for i, feat in enumerate(engineered_features[:10], 1):  # Show first 10
        summary.append(f"   {i}. {feat}")
    if len(engineered_features) > 10:
        summary.append(f"   ... and {len(engineered_features) - 10} more")
    
    summary.append(f"\n7. NEXT STEPS:")
    summary.append(f"   - Proceed to Task 2: Model Building")
    summary.append(f"   - Use processed data in data/processed/")
    summary.append(f"   - Apply SMOTE during model training")
    
    # Write to file
    with open('reports/task1_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    # Also print to console
    print('\n'.join(summary))

if __name__ == "__main__":
    main()