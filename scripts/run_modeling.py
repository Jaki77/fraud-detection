"""
Main script to execute Task 2: Model Building and Training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.model_training import FraudDetectionModel
from src.evaluation import ModelEvaluator
from src.visualization import ModelComparisonVisualizer
import joblib

def main():
    print("="*70)
    print("TASK 2: MODEL BUILDING AND TRAINING")
    print("="*70)
    
    # Create directories if they don't exist
    os.makedirs('reports/figures/model_performance', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load processed data
    print("\n1. Loading processed data...")
    
    fraud_data = pd.read_csv('data/processed/fraud_data_engineered.csv')
    credit_data = pd.read_csv('data/processed/creditcard_processed.csv')
    
    print(f"✅ Fraud Data: {fraud_data.shape}")
    print(f"✅ Credit Data: {credit_data.shape}")
    
    # Step 2: Prepare e-commerce data
    print("\n2. Preparing e-commerce data for modeling...")
    
    # Exclude identifiers and timestamps
    excluded_features = ['user_id', 'signup_time', 'purchase_time', 'ip_address']
    X_ecomm = fraud_data.drop(['class'] + excluded_features, axis=1, errors='ignore')
    y_ecomm = fraud_data['class']
    
    # One-hot encode categorical variables
    categorical_cols = X_ecomm.select_dtypes(include=['object', 'category']).columns
    X_ecomm_encoded = pd.get_dummies(X_ecomm, columns=categorical_cols, drop_first=True)
    
    print(f"✅ Features: {X_ecomm_encoded.shape[1]}")
    print(f"✅ Target distribution - No Fraud: {(y_ecomm == 0).sum():,}, Fraud: {(y_ecomm == 1).sum():,}")
    
    # Step 3: Train models
    print("\n3. Training models...")
    
    model_trainer = FraudDetectionModel(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(
        X_ecomm_encoded, y_ecomm, test_size=0.2, apply_smote=True
    )
    
    # Train models
    print("\nTraining Logistic Regression...")
    logistic_model, logistic_results = model_trainer.train_baseline_logistic(
        X_train, X_test, y_train, y_test
    )
    
    print("\nTraining Random Forest...")
    rf_model, rf_results = model_trainer.train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    print("\nTraining XGBoost...")
    xgb_model, xgb_results = model_trainer.train_xgboost(
        X_train, X_test, y_train, y_test
    )
    
    # Step 4: Model comparison
    print("\n4. Comparing models...")
    
    comparison_df = model_trainer.compare_models()
    
    # Create visual comparison
    visualizer = ModelComparisonVisualizer()
    
    # Plot comparison
    fig_comparison = visualizer.plot_model_comparison(comparison_df, "Model Performance Comparison")
    fig_comparison.write_image("reports/figures/model_performance/model_comparison.png")
    fig_comparison.write_html("reports/figures/model_performance/model_comparison.html")
    
    # Plot confusion matrices
    models_results = [logistic_results, rf_results, xgb_results]
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    fig_cm = visualizer.plot_confusion_matrices_side_by_side(models_results, model_names)
    fig_cm.write_image("reports/figures/model_performance/confusion_matrices_comparison.png")
    
    # Plot ROC curves
    fig_roc = visualizer.plot_roc_curves_comparison(models_results, model_names)
    fig_roc.write_image("reports/figures/model_performance/roc_curves_comparison.png")
    
    # Plot PR curves
    fig_pr = visualizer.plot_pr_curves_comparison(models_results, model_names)
    fig_pr.write_image("reports/figures/model_performance/pr_curves_comparison.png")
    
    # Step 5: Save models and results
    print("\n5. Saving models and results...")
    
    # Save models
    model_trainer.save_model(logistic_model, 'logistic_regression_ecomm')
    model_trainer.save_model(rf_model, 'random_forest_ecomm')
    model_trainer.save_model(xgb_model, 'xgboost_ecomm')
    
    # Save feature names
    feature_names = X_ecomm_encoded.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names_ecomm.pkl')
    
    # Save results summary
    results_summary = {
        'logistic_regression': logistic_results,
        'random_forest': rf_results,
        'xgboost': xgb_results,
        'comparison': comparison_df.to_dict()
    }
    
    joblib.dump(results_summary, 'models/model_results_summary.pkl')
    
    # Step 6: Generate report
    print("\n6. Generating model selection report...")
    
    generate_model_selection_report(comparison_df, models_results, model_names)
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Outputs generated:")
    print("  ✅ Trained models saved to 'models/'")
    print("  ✅ Performance visualizations saved to 'reports/figures/model_performance/'")
    print("  ✅ Model comparison completed")
    print("  ✅ Model selection report generated")

def generate_model_selection_report(comparison_df, models_results, model_names):
    """Generate a detailed model selection report."""
    
    report = []
    report.append("="*70)
    report.append("MODEL SELECTION REPORT")
    report.append("="*70)
    
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find best model based on AUC-PR
    best_model_idx = comparison_df['AUC-PR'].idxmax()
    best_model_name = best_model_idx
    best_auc_pr = comparison_df.loc[best_model_idx, 'AUC-PR']
    
    report.append(f"\n🏆 SELECTED MODEL: {best_model_name}")
    report.append(f"   Selection Criteria: Highest AUC-PR ({best_auc_pr:.4f})")
    report.append(f"   Reason: AUC-PR is the most important metric for imbalanced fraud detection")
    
    # Model performance summary
    report.append(f"\n📈 PERFORMANCE SUMMARY:")
    for model_name in model_names:
        if model_name in comparison_df.index:
            row = comparison_df.loc[model_name]
            report.append(f"\n  {model_name}:")
            report.append(f"    AUC-PR:      {row['AUC-PR']:.4f}")
            report.append(f"    F1-Score:    {row['F1-Score']:.4f}")
            report.append(f"    Precision:   {row['Precision']:.4f}")
            report.append(f"    Recall:      {row['Recall']:.4f}")
    
    # Business impact analysis
    report.append(f"\n💼 BUSINESS IMPACT ANALYSIS:")
    report.append(f"  Metrics considered for fraud detection:")
    report.append(f"  1. False Positive Rate (Customer friction)")
    report.append(f"  2. False Negative Rate (Fraud losses)")
    report.append(f"  3. Precision (Accuracy of fraud alerts)")
    report.append(f"  4. Recall (Coverage of actual fraud)")
    
    # Model strengths and weaknesses
    report.append(f"\n🔍 MODEL CHARACTERISTICS:")
    report.append(f"\n  Logistic Regression:")
    report.append(f"    ✓ Pros: Highly interpretable, fast training")
    report.append(f"    ✗ Cons: May not capture complex patterns")
    
    report.append(f"\n  Random Forest:")
    report.append(f"    ✓ Pros: Handles non-linear patterns, robust to outliers")
    report.append(f"    ✗ Cons: Less interpretable, can overfit")
    
    report.append(f"\n  XGBoost:")
    report.append(f"    ✓ Pros: State-of-the-art performance, handles imbalance well")
    report.append(f"    ✗ Cons: Complex tuning, slower training")
    
    # Deployment recommendations
    report.append(f"\n🚀 DEPLOYMENT RECOMMENDATIONS:")
    report.append(f"  1. Start with {best_model_name} as primary model")
    report.append(f"  2. Use Logistic Regression for model auditing and explainability")
    report.append(f"  3. Implement ensemble approach in production")
    report.append(f"  4. Monitor false positive rate to maintain customer experience")
    
    # Save report
    with open('reports/model_selection_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))
    
    return report

if __name__ == "__main__":
    main()