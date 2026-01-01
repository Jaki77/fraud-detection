import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """Comprehensive SHAP analysis for model explainability."""
    
    def __init__(self, model, X, feature_names=None, model_type='tree'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X: Feature matrix (for background samples)
            feature_names: List of feature names
            model_type: 'tree' for tree-based, 'linear' for linear models
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.model_type = model_type
        
        # Initialize appropriate SHAP explainer
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X)
        else:
            self.explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get SHAP values for positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class (fraud)
        
        print(f"SHAP values shape: {self.shap_values.shape}")
    
    def global_feature_importance(self):
        """Calculate global feature importance from SHAP values."""
        # Mean absolute SHAP values
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, max_features=20, save_path=None):
        """
        Create SHAP summary plot (beeswarm plot).
        
        Args:
            max_features: Maximum number of features to display
            save_path: Path to save the plot
        """
        print("\nGenerating SHAP summary plot...")
        
        # Get top features
        importance_df = self.global_feature_importance()
        top_features = importance_df.head(max_features)['feature'].tolist()
        top_indices = [self.feature_names.index(f) for f in top_features]
        
        # Filter data for top features
        X_top = pd.DataFrame(self.X[:, top_indices], columns=top_features)
        shap_values_top = self.shap_values[:, top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values_top, 
            X_top,
            feature_names=top_features,
            max_display=max_features,
            show=False,
            plot_size=(12, 8)
        )
        
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {save_path}")
        
        plt.show()
        
        return importance_df
    
    def plot_waterfall(self, instance_idx, max_features=10, save_path=None):
        """
        Create SHAP waterfall plot for individual prediction.
        
        Args:
            instance_idx: Index of instance to explain
            max_features: Maximum features to display
            save_path: Path to save the plot
        """
        print(f"\nGenerating waterfall plot for instance {instance_idx}...")
        
        # Get SHAP values for this instance
        shap_value = self.explainer.shap_values(self.X[instance_idx:instance_idx+1])
        if isinstance(shap_value, list):
            shap_value = shap_value[1][0]  # Positive class
        
        # Create waterfall plot
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_value,
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                          else self.explainer.expected_value,
                feature_names=self.feature_names,
                data=self.X[instance_idx]
            ),
            max_display=max_features,
            show=False
        )
        
        plt.title(f"Waterfall Plot - Instance {instance_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to: {save_path}")
        
        plt.show()
        
        return {
            'instance_idx': instance_idx,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                         else self.explainer.expected_value,
            'shap_values': shap_value,
            'prediction': self.model.predict(self.X[instance_idx:instance_idx+1])[0],
            'prediction_proba': self.model.predict_proba(self.X[instance_idx:instance_idx+1])[0][1]
        }
    
    def plot_force(self, instance_idx, max_features=10, save_path=None):
        """
        Create SHAP force plot for individual prediction.
        
        Args:
            instance_idx: Index of instance to explain
            max_features: Maximum features to display
            save_path: Path to save the plot (HTML format)
        """
        print(f"\nGenerating force plot for instance {instance_idx}...")
        
        # Get SHAP values
        shap_value = self.explainer.shap_values(self.X[instance_idx:instance_idx+1])
        if isinstance(shap_value, list):
            shap_value = shap_value[1]
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
            else self.explainer.expected_value,
            shap_value[0],
            self.X[instance_idx:instance_idx+1],
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        # Save as HTML
        if save_path:
            shap.save_html(save_path, force_plot)
            print(f"Force plot saved to: {save_path}")
        
        return force_plot
    
    def plot_dependence(self, feature_name, interaction_feature=None, save_path=None):
        """
        Create SHAP dependence plot.
        
        Args:
            feature_name: Main feature to analyze
            interaction_feature: Feature for color coding
            save_path: Path to save the plot
        """
        print(f"\nGenerating dependence plot for {feature_name}...")
        
        feature_idx = self.feature_names.index(feature_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.X,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                ax=ax,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.X,
                feature_names=self.feature_names,
                ax=ax,
                show=False
            )
        
        title = f"SHAP Dependence Plot: {feature_name}"
        if interaction_feature:
            title += f" vs {interaction_feature}"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dependence plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, y_true, y_pred):
        """
        Analyze different types of predictions for SHAP analysis.
        
        Returns indices of:
        - True Positives (correct fraud detection)
        - False Positives (legitimate flagged as fraud)
        - False Negatives (missed fraud cases)
        """
        results = {}
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # True Positives
        tp_mask = (y_true == 1) & (y_pred == 1)
        results['true_positives'] = np.where(tp_mask)[0]
        
        # False Positives
        fp_mask = (y_true == 0) & (y_pred == 1)
        results['false_positives'] = np.where(fp_mask)[0]
        
        # False Negatives
        fn_mask = (y_true == 1) & (y_pred == 0)
        results['false_negatives'] = np.where(fn_mask)[0]
        
        print(f"True Positives: {len(results['true_positives'])}")
        print(f"False Positives: {len(results['false_positives'])}")
        print(f"False Negatives: {len(results['false_negatives'])}")
        
        return results
    
    def create_comparative_analysis(self, prediction_types, n_examples=3):
        """
        Create comparative SHAP analysis for different prediction types.
        
        Args:
            prediction_types: Dictionary from analyze_predictions()
            n_examples: Number of examples to analyze per type
        """
        analysis_results = {}
        
        for pred_type, indices in prediction_types.items():
            if len(indices) > 0:
                print(f"\nAnalyzing {pred_type.replace('_', ' ').title()}...")
                examples = indices[:min(n_examples, len(indices))]
                
                type_results = []
                for idx in examples:
                    result = self.plot_waterfall(idx, max_features=10, save_path=None)
                    type_results.append(result)
                
                analysis_results[pred_type] = type_results
        
        return analysis_results
    
    def generate_interactive_dashboard(self, save_path='reports/shap_dashboard.html'):
        """Generate an interactive SHAP dashboard."""
        print("\nGenerating interactive SHAP dashboard...")
        
        # Get feature importance
        importance_df = self.global_feature_importance()
        top_features = importance_df.head(15)
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 15 Feature Importance (SHAP)',
                'SHAP Value Distribution',
                'Feature Value vs SHAP Value',
                'Prediction Distribution'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'box'}],
                [{'type': 'scatter'}, {'type': 'histogram'}]
            ]
        )
        
        # 1. Feature importance bar chart
        fig.add_trace(
            go.Bar(
                x=top_features['shap_importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='steelblue',
                name='SHAP Importance'
            ),
            row=1, col=1
        )
        
        # 2. SHAP value distribution for top features
        for i, feature in enumerate(top_features['feature'].head(5)):
            feature_idx = self.feature_names.index(feature)
            fig.add_trace(
                go.Box(
                    y=self.shap_values[:, feature_idx],
                    name=feature,
                    boxpoints='outliers',
                    marker_color=plt.cm.Set1(i/5)
                ),
                row=1, col=2
            )
        
        # 3. Scatter plot for top feature
        if len(top_features) > 0:
            top_feature = top_features.iloc[0]['feature']
            top_idx = self.feature_names.index(top_feature)
            
            fig.add_trace(
                go.Scatter(
                    x=self.X[:, top_idx],
                    y=self.shap_values[:, top_idx],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.6,
                        color=self.shap_values[:, top_idx],
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title='SHAP Value')
                    ),
                    name=top_feature
                ),
                row=2, col=1
            )
        
        # 4. Prediction probability distribution
        predictions = self.model.predict_proba(self.X)[:, 1]
        fig.add_trace(
            go.Histogram(
                x=predictions,
                nbinsx=50,
                name='Prediction Probability',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="SHAP Analysis Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="SHAP Importance", row=1, col=1)
        fig.update_xaxes(title_text="SHAP Value", row=1, col=2)
        fig.update_xaxes(title_text=top_feature if len(top_features) > 0 else "Feature Value", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Probability", row=2, col=2)
        
        fig.update_yaxes(title_text="Feature", row=1, col=1)
        fig.update_yaxes(title_text="SHAP Value", row=1, col=2)
        fig.update_yaxes(title_text="SHAP Value", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Save dashboard
        fig.write_html(save_path)
        print(f"Dashboard saved to: {save_path}")
        
        return fig