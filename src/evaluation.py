# src/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, 
    precision_score, recall_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Evaluate model performance with appropriate metrics for imbalanced data."""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
    def evaluate(self, model_name=""):
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        results = {}
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        results['confusion_matrix'] = cm
        
        # 2. Classification Metrics
        results['precision'] = precision_score(self.y_test, self.y_pred)
        results['recall'] = recall_score(self.y_test, self.y_pred)
        results['f1'] = f1_score(self.y_test, self.y_pred)
        
        # 3. AUC-ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        results['auc_roc'] = auc(fpr, tpr)
        results['roc_curve'] = (fpr, tpr)
        
        # 4. AUC-PR (More important for imbalanced data)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        results['auc_pr'] = auc(recall, precision)
        results['pr_curve'] = (precision, recall)
        results['avg_precision'] = average_precision_score(self.y_test, self.y_pred_proba)
        
        # 5. Business-oriented metrics
        tn, fp, fn, tp = cm.ravel()
        results['false_positive_rate'] = fp / (fp + tn)
        results['false_negative_rate'] = fn / (fn + tp)
        results['true_positive_rate'] = tp / (tp + fn)
        results['true_negative_rate'] = tn / (tn + fp)
        
        # Cost analysis (assuming business costs)
        # False positive cost: customer dissatisfaction, manual review cost
        # False negative cost: actual fraud loss
        false_positive_cost = 50  # Example: $50 per false alarm
        false_negative_cost = 500  # Example: $500 per missed fraud
        results['estimated_cost'] = (fp * false_positive_cost) + (fn * false_negative_cost)
        
        # Print summary
        self._print_summary(results, model_name)
        
        # Generate visualizations
        self.plot_evaluation_metrics(results, model_name)
        
        return results
    
    def _print_summary(self, results, model_name):
        """Print evaluation summary."""
        print("\n" + "-"*40)
        print(f"EVALUATION SUMMARY: {model_name}")
        print("-"*40)
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                No Fraud | Fraud")
        print(f"Actual No Fraud   {tn:6}  | {fp:5}")
        print(f"Actual Fraud      {fn:6}  | {tp:5}")
        
        # Key metrics
        print(f"\nKey Metrics:")
        print(f"Precision (Fraud):    {results['precision']:.4f}")
        print(f"Recall (Fraud):       {results['recall']:.4f}")
        print(f"F1-Score:            {results['f1']:.4f}")
        print(f"AUC-ROC:             {results['auc_roc']:.4f}")
        print(f"AUC-PR:              {results['auc_pr']:.4f}")
        print(f"Avg Precision:       {results['avg_precision']:.4f}")
        
        print(f"\nBusiness Impact:")
        print(f"False Positive Rate: {results['false_positive_rate']:.4%}")
        print(f"False Negative Rate: {results['false_negative_rate']:.4%}")
        print(f"Estimated Cost:     ${results['estimated_cost']:,.2f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['No Fraud', 'Fraud']))
    
    def plot_evaluation_metrics(self, results, model_name=""):
        """Generate comprehensive evaluation plots."""
        print("\nGenerating evaluation plots...")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 
                           'Precision-Recall Curve', 'Feature Importance'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. Confusion Matrix Heatmap
        cm = results['confusion_matrix']
        labels = ['No Fraud', 'Fraud']
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve
        fpr, tpr = results['roc_curve']
        auc_roc = results['auc_roc']
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc_roc:.3f})',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        
        # 3. Precision-Recall Curve
        precision, recall = results['pr_curve']
        auc_pr = results['auc_pr']
        
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AUC = {auc_pr:.3f})',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        
        # 4. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X_test.columns if hasattr(self.X_test, 'columns') else \
                           [f'Feature_{i}' for i in range(len(importances))]
            
            # Sort by importance
            indices = np.argsort(importances)[-10:]  # Top 10 features
            fig.add_trace(
                go.Bar(
                    x=importances[indices],
                    y=[feature_names[i] for i in indices],
                    orientation='h',
                    marker_color='coral'
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Importance", row=2, col=2)
        else:
            # If no feature importance, show prediction distribution
            fig.add_trace(
                go.Histogram(
                    x=self.y_pred_proba[self.y_test == 0],
                    name='No Fraud',
                    opacity=0.7,
                    nbinsx=50
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Histogram(
                    x=self.y_pred_proba[self.y_test == 1],
                    name='Fraud',
                    opacity=0.7,
                    nbinsx=50
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Predicted Probability", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Model Evaluation: {model_name}",
            showlegend=True,
            template='plotly_white'
        )
        
        # Save figure
        fig.write_image(f"reports/figures/model_performance/{model_name.replace(' ', '_')}_evaluation.png")
        fig.write_html(f"reports/figures/model_performance/{model_name.replace(' ', '_')}_evaluation.html")
        
        print(f"Plots saved to reports/figures/model_performance/")
        
        return fig
    
    def get_detailed_analysis(self):
        """
        Get detailed analysis of predictions.
        """
        # Create results DataFrame
        results_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.y_pred,
            'probability': self.y_pred_proba
        })
        
        # Add indices for merging with original data if needed
        if hasattr(self.X_test, 'index'):
            results_df.index = self.X_test.index
        
        # Categorize predictions
        results_df['prediction_type'] = 'Correct Non-Fraud'
        results_df.loc[(results_df['actual'] == 1) & (results_df['predicted'] == 1), 'prediction_type'] = 'True Positive'
        results_df.loc[(results_df['actual'] == 0) & (results_df['predicted'] == 1), 'prediction_type'] = 'False Positive'
        results_df.loc[(results_df['actual'] == 1) & (results_df['predicted'] == 0), 'prediction_type'] = 'False Negative'
        
        return results_df