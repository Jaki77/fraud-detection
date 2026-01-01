"""
Main script to execute Task 3: Model Explainability with SHAP.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
from src.explainability import SHAPExplainer
from src.business_recommendations import BusinessRecommendations

def main():
    print("="*70)
    print("TASK 3: MODEL EXPLAINABILITY WITH SHAP")
    print("="*70)
    
    # Create directories if they don't exist
    os.makedirs('reports/figures/shap_plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Step 1: Load data and model
    print("\n1. Loading data and model...")
    
    # Load best model (assuming XGBoost from Task 2)
    try:
        model = joblib.load('models/xgboost_ecomm.pkl')
        print(f"✅ Model loaded: {type(model).__name__}")
    except:
        print("⚠️  XGBoost model not found, trying Random Forest...")
        model = joblib.load('models/random_forest_ecomm.pkl')
        print(f"✅ Model loaded: {type(model).__name__}")
    
    # Load data
    fraud_data = pd.read_csv('data/processed/fraud_data_engineered.csv')
    feature_names = joblib.load('models/feature_names_ecomm.pkl')
    
    # Prepare features
    excluded_features = ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'class']
    X = fraud_data.drop(excluded_features, axis=1, errors='ignore')
    
    # One-hot encode
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Align with training features
    missing_cols = set(feature_names) - set(X_encoded.columns)
    for col in missing_cols:
        X_encoded[col] = 0
    X_encoded = X_encoded[feature_names]
    
    y = fraud_data['class']
    
    # Use sample for faster computation
    sample_size = min(2000, len(X_encoded))
    sample_indices = np.random.choice(len(X_encoded), sample_size, replace=False)
    X_sample = X_encoded.iloc[sample_indices].values
    y_sample = y.iloc[sample_indices].values
    
    print(f"✅ Using {sample_size} samples for SHAP analysis")
    
    # Step 2: Initialize SHAP explainer
    print("\n2. Initializing SHAP explainer...")
    
    shap_explainer = SHAPExplainer(
        model=model,
        X=X_sample,
        feature_names=feature_names,
        model_type='tree'
    )
    
    # Step 3: Global feature importance
    print("\n3. Calculating global feature importance...")
    
    shap_importance = shap_explainer.global_feature_importance()
    
    print("\nTop 10 Most Important Features:")
    print("-"*40)
    for i, (_, row) in enumerate(shap_importance.head(10).iterrows(), 1):
        print(f"{i:2}. {row['feature']:30} {row['shap_importance']:.4f}")
    
    # Step 4: Generate SHAP plots
    print("\n4. Generating SHAP visualizations...")
    
    # Summary plot
    shap_explainer.plot_summary(
        max_features=15,
        save_path='reports/figures/shap_plots/shap_summary.png'
    )
    
    # Step 5: Individual prediction analysis
    print("\n5. Analyzing individual predictions...")
    
    # Get predictions
    y_pred = model.predict(X_sample)
    
    # Analyze prediction types
    prediction_types = shap_explainer.analyze_predictions(y_sample, y_pred)
    
    # Analyze examples
    analysis_examples = {}
    
    # True Positive
    if len(prediction_types['true_positives']) > 0:
        tp_idx = prediction_types['true_positives'][0]
        analysis_examples['true_positive'] = shap_explainer.plot_waterfall(
            tp_idx,
            max_features=10,
            save_path='reports/figures/shap_plots/true_positive_waterfall.png'
        )
        print(f"✅ True Positive analysis saved")
    
    # False Positive
    if len(prediction_types['false_positives']) > 0:
        fp_idx = prediction_types['false_positives'][0]
        analysis_examples['false_positive'] = shap_explainer.plot_waterfall(
            fp_idx,
            max_features=10,
            save_path='reports/figures/shap_plots/false_positive_waterfall.png'
        )
        print(f"✅ False Positive analysis saved")
    
    # False Negative
    if len(prediction_types['false_negatives']) > 0:
        fn_idx = prediction_types['false_negatives'][0]
        analysis_examples['false_negative'] = shap_explainer.plot_waterfall(
            fn_idx,
            max_features=10,
            save_path='reports/figures/shap_plots/false_negative_waterfall.png'
        )
        print(f"✅ False Negative analysis saved")
    
    # Step 6: Generate business recommendations
    print("\n6. Generating business recommendations...")
    
    # Load model performance
    model_results = joblib.load('models/model_results_summary.pkl')
    best_model_performance = model_results.get('xgboost', model_results.get('random_forest', {}))
    
    # Feature descriptions
    feature_descriptions = {
        'time_since_signup': 'Hours since user account creation',
        'purchase_value': 'Transaction amount in dollars',
        'transactions_last_1h': 'Number of transactions in last 1 hour',
        'transactions_last_24h': 'Number of transactions in last 24 hours',
        'hour_of_day': 'Hour of day when transaction occurred',
        'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
        'is_weekend': 'Transaction occurred on weekend',
        'users_per_device': 'Number of users sharing same device'
    }
    
    business_analyst = BusinessRecommendations(shap_importance, feature_descriptions)
    
    recommendations = business_analyst.generate_recommendations(
        shap_analysis_results=shap_explainer,
        model_performance=best_model_performance,
        n_recommendations=5
    )
    
    roadmap = business_analyst.generate_implementation_roadmap(recommendations)
    
    # Step 7: Generate reports
    print("\n7. Generating reports...")
    
    # Save SHAP importance
    shap_importance.to_csv('reports/shap_feature_importance.csv', index=False)
    
    # Save recommendations
    recommendations.to_csv('reports/business_recommendations.csv', index=False)
    roadmap.to_csv('reports/implementation_roadmap.csv', index=False)
    
    # Generate executive summary
    top_drivers = business_analyst.get_top_drivers(10)
    executive_summary = business_analyst.create_executive_summary(
        top_drivers,
        recommendations,
        best_model_performance
    )
    
    with open('reports/executive_summary.txt', 'w') as f:
        f.write(executive_summary)
    
    # Step 8: Generate interactive dashboard
    print("\n8. Creating interactive dashboard...")
    
    shap_explainer.generate_interactive_dashboard(
        save_path='reports/figures/shap_plots/shap_dashboard.html'
    )
    
    print("\n" + "="*70)
    print("TASK 3 COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Outputs generated:")
    print("  ✅ SHAP feature importance analysis")
    print("  ✅ Individual prediction explanations (True/False Positives/Negatives)")
    print("  ✅ 5 actionable business recommendations")
    print("  ✅ Implementation roadmap")
    print("  ✅ Executive summary report")
    print("  ✅ All visualizations saved to 'reports/figures/shap_plots/'")

if __name__ == "__main__":
    main()