import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ModelComparisonVisualizer:
    """Visualize model comparison results."""
    
    @staticmethod
    def plot_model_comparison(comparison_df, title="Model Comparison"):
        """Create comparison bar chart for models."""
        fig = go.Figure()
        
        metrics = ['AUC-PR', 'F1-Score', 'Precision', 'Recall', 'Avg Precision']
        
        for metric in metrics:
            if metric in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df.index,
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition='auto',
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_confusion_matrices_side_by_side(models_results, model_names):
        """Plot confusion matrices for multiple models."""
        n_models = len(models_results)
        
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=model_names,
            horizontal_spacing=0.1
        )
        
        for i, (model_name, results) in enumerate(zip(model_names, models_results), 1):
            cm = results['confusion_matrix']
            
            heatmap = go.Heatmap(
                z=cm,
                x=['Predicted No', 'Predicted Yes'],
                y=['Actual No', 'Actual Yes'],
                text=[[f"{val:,}" for val in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale='Blues',
                showscale=False if i < n_models else True
            )
            
            fig.add_trace(heatmap, row=1, col=i)
        
        fig.update_layout(
            title="Confusion Matrices Comparison",
            height=400,
            width=300 * n_models
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curves_comparison(models_results, model_names):
        """Plot ROC curves for multiple models."""
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False
        ))
        
        # Add each model's ROC curve
        for model_name, results in zip(model_names, models_results):
            fpr, tpr = results['roc_curve']
            auc_score = results['auc_roc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_pr_curves_comparison(models_results, model_names):
        """Plot Precision-Recall curves for multiple models."""
        fig = go.Figure()
        
        for model_name, results in zip(model_names, models_results):
            precision, recall = results['pr_curve']
            auc_pr = results['auc_pr']
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AUC = {auc_pr:.3f})',
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Precision-Recall Curves Comparison",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template='plotly_white',
            height=500
        )
        
        return fig