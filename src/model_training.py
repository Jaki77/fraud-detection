import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

class FraudDetectionModel:
    """Build and train fraud detection models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X, y, test_size=0.2, apply_smote=True):
        """
        Prepare data for modeling with train-test split and optional SMOTE.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            apply_smote: Whether to apply SMOTE to training data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Preparing data for modeling...")
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Fraud rate in training: {(y_train == 1).mean():.4%}")
        print(f"Fraud rate in test: {(y_test == 1).mean():.4%}")
        
        if apply_smote:
            print("\nApplying SMOTE to training data...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"Training set after SMOTE: {X_train.shape[0]} samples")
            print(f"Class distribution after SMOTE: {np.unique(y_train, return_counts=True)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_logistic(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Train Logistic Regression as baseline model.
        
        Returns:
            Trained model and evaluation results
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE: LOGISTIC REGRESSION")
        print("="*60)
        
        # Default parameters
        params = {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'solver': 'lbfgs'
        }
        params.update(kwargs)
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Store model
        self.models['logistic'] = model
        
        # Evaluate
        from src.evaluation import ModelEvaluator
        evaluator = ModelEvaluator(model, X_test, y_test)
        results = evaluator.evaluate('Logistic Regression')
        
        self.results['logistic'] = results
        return model, results
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Train Random Forest classifier.
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        # Default parameters optimized for fraud detection
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': 'balanced_subsample',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        # Evaluate
        from src.evaluation import ModelEvaluator
        evaluator = ModelEvaluator(model, X_test, y_test)
        results = evaluator.evaluate('Random Forest')
        
        self.results['random_forest'] = results
        return model, results
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Train XGBoost classifier.
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        # Default parameters optimized for imbalanced data
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
            'random_state': self.random_state,
            'eval_metric': 'aucpr',  # Use AUC-PR for imbalanced data
            'use_label_encoder': False,
            'verbosity': 0
        }
        params.update(kwargs)
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        # Evaluate
        from src.evaluation import ModelEvaluator
        evaluator = ModelEvaluator(model, X_test, y_test)
        results = evaluator.evaluate('XGBoost')
        
        self.results['xgboost'] = results
        return model, results
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Train LightGBM classifier.
        """
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM")
        print("="*60)
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'is_unbalance': True,
            'random_state': self.random_state,
            'verbose': -1
        }
        params.update(kwargs)
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        
        # Evaluate
        from src.evaluation import ModelEvaluator
        evaluator = ModelEvaluator(model, X_test, y_test)
        results = evaluator.evaluate('LightGBM')
        
        self.results['lightgbm'] = results
        return model, results
    
    def perform_cross_validation(self, model, X, y, cv=5, model_name="Model"):
        """
        Perform stratified k-fold cross-validation.
        """
        print(f"\nPerforming {cv}-fold Stratified Cross-Validation for {model_name}...")
        
        cv_scores_auc = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        cv_scores_f1 = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1
        )
        
        print(f"AUC-ROC CV Scores: {cv_scores_auc}")
        print(f"AUC-ROC Mean: {cv_scores_auc.mean():.4f}, Std: {cv_scores_auc.std():.4f}")
        print(f"F1-Score CV Scores: {cv_scores_f1}")
        print(f"F1-Score Mean: {cv_scores_f1.mean():.4f}, Std: {cv_scores_f1.std():.4f}")
        
        return {
            'auc_scores': cv_scores_auc,
            'f1_scores': cv_scores_f1,
            'auc_mean': cv_scores_auc.mean(),
            'auc_std': cv_scores_auc.std(),
            'f1_mean': cv_scores_f1.mean(),
            'f1_std': cv_scores_f1.std()
        }
    
    def compare_models(self):
        """
        Compare performance of all trained models.
        """
        if not self.results:
            print("No models have been trained yet.")
            return None
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame()
        
        for model_name, results in self.results.items():
            comparison_df[model_name] = pd.Series({
                'AUC-PR': results.get('auc_pr', np.nan),
                'F1-Score': results.get('f1', np.nan),
                'Precision': results.get('precision', np.nan),
                'Recall': results.get('recall', np.nan),
                'Avg Precision': results.get('avg_precision', np.nan)
            })
        
        # Transpose for better readability
        comparison_df = comparison_df.T
        
        print("\nPerformance Comparison:")
        print(comparison_df.round(4))
        
        # Identify best model based on AUC-PR (most important for imbalanced data)
        best_model = comparison_df['AUC-PR'].idxmax()
        best_score = comparison_df.loc[best_model, 'AUC-PR']
        
        print(f"\n🏆 Best Model: {best_model} (AUC-PR: {best_score:.4f})")
        
        return comparison_df
    
    def save_model(self, model, filename):
        """Save trained model to disk."""
        joblib.dump(model, f'models/{filename}.pkl')
        print(f"Model saved to: models/{filename}.pkl")
    
    def load_model(self, filename):
        """Load trained model from disk."""
        model = joblib.load(f'models/{filename}.pkl')
        print(f"Model loaded from: models/{filename}.pkl")
        return model