# gbm_models.py - Gradient Boosted Models for Telco Churn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from prep import get_preprocessed_data

class GBMModels:
    """Class for training and evaluating Gradient Boosted Models"""
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_gradient_boosting(self, **params):
        """Train Sklearn Gradient Boosting Classifier"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        default_params.update(params)
        
        print("Training Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['GradientBoosting'] = model
        self.evaluate_model(model, 'GradientBoosting')
        
        return model

    def train_xgboost(self, **params):
        """Train XGBoost Classifier"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        default_params.update(params)
        
        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(**default_params)
        
        # ✅ Fixed for XGBoost >= 2.0 (use callbacks instead of early_stopping_rounds)
        eval_set = [(self.X_val, self.y_val)]
        model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.models['XGBoost'] = model
        self.evaluate_model(model, 'XGBoost')
        
        return model
    
    def train_lightgbm(self, **params):
        """Train LightGBM Classifier"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(params)
        
        print("Training LightGBM Classifier...")
        model = lgb.LGBMClassifier(**default_params)
        
        # Train with early stopping
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        self.models['LightGBM'] = model
        self.evaluate_model(model, 'LightGBM')
        
        return model
    
    def train_adaboost(self, **params):
        """Train AdaBoost Classifier"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 1.0,
            'random_state': 42
        }
        default_params.update(params)
        
        print("Training AdaBoost Classifier...")
        model = AdaBoostClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['AdaBoost'] = model
        self.evaluate_model(model, 'AdaBoost')
        
        return model
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        # Validation set performance
        val_pred = model.predict(self.X_val)
        val_prob = model.predict_proba(self.X_val)[:, 1]
        
        val_metrics = self.calculate_metrics(self.y_val, val_pred, val_prob)
        
        print(f"\n{model_name} Validation Performance:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        
        self.results[model_name] = {'val_metrics': val_metrics}
        
        return val_metrics
    
    def hyperparameter_tuning_xgboost(self):
        """Comprehensive hyperparameter tuning for XGBoost"""
        print("\nPerforming XGBoost Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, 
            n_iter=50,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print("Best XGBoost Parameters:", random_search.best_params_)
        print("Best CV Score:", random_search.best_score_)
        
        best_model = random_search.best_estimator_
        self.models['XGBoost_Tuned'] = best_model
        self.evaluate_model(best_model, 'XGBoost_Tuned')
        
        return best_model, random_search.best_params_
    
    def hyperparameter_tuning_lightgbm(self):
        """Comprehensive hyperparameter tuning for LightGBM"""
        print("\nPerforming LightGBM Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'num_leaves': [20, 31, 40, 50],
            'max_depth': [-1, 5, 10, 15],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            random_state=42,
            verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            lgb_model, param_grid,
            n_iter=50,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print("Best LightGBM Parameters:", random_search.best_params_)
        print("Best CV Score:", random_search.best_score_)
        
        best_model = random_search.best_estimator_
        self.models['LightGBM_Tuned'] = best_model
        self.evaluate_model(best_model, 'LightGBM_Tuned')
        
        return best_model, random_search.best_params_
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def plot_feature_importance_comparison(self, top_n=20):
        """Compare feature importance across all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        model_names = ['GradientBoosting', 'XGBoost_Tuned', 'LightGBM_Tuned', 'AdaBoost']
        
        for idx, model_name in enumerate(model_names):
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    continue
                
                indices = np.argsort(importances)[::-1][:top_n]
                
                axes[idx].barh(range(top_n), importances[indices])
                axes[idx].set_yticks(range(top_n))
                axes[idx].set_yticklabels([self.X_train.columns[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name} - Top {top_n} Features')
        
        plt.tight_layout()
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/gbm_feature_importance_comparison.png")
        plt.show()
    
    def plot_learning_curves(self):
        """Plot learning curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for idx, (model_name, model) in enumerate(list(self.models.items())[:4]):
            train_scores = []
            val_scores = []
            
            for train_size in train_sizes:
                n_samples = int(train_size * len(self.X_train))
                X_subset = self.X_train[:n_samples]
                y_subset = self.y_train[:n_samples]
                
                # Clone and train model
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_subset, y_subset)
                
                # Score
                train_prob = model_clone.predict_proba(X_subset)[:, 1]
                val_prob = model_clone.predict_proba(self.X_val)[:, 1]
                
                train_scores.append(roc_auc_score(y_subset, train_prob))
                val_scores.append(roc_auc_score(self.y_val, val_prob))
            
            axes[idx].plot(train_sizes, train_scores, 'o-', label='Training')
            axes[idx].plot(train_sizes, val_scores, 'o-', label='Validation')
            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('AUC Score')
            axes[idx].set_title(f'{model_name} Learning Curve')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        plt.savefig(f"{output_dir}/gbm_learning_curves.png")
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_score = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Gradient Boosted Models', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        plt.savefig(f"{output_dir}/gbm_roc_curves.png")
        plt.show()
    
    def final_evaluation(self):
        """Perform final evaluation on test set"""
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            metrics = self.calculate_metrics(self.y_test, y_pred, y_prob)
            
            results.append({
                'Model': model_name,
                'AUC': metrics['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1']
            })
            
            print(f"\n{model_name}:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AUC', ascending=False)
        
        return results_df
    
    def save_models(self, output_dir=None):
        """Save all trained models"""
        if output_dir is None:
            output_dir = "/Users/ob/Projects/teleco/models"
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f"gbm_{model_name.lower()}.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")

def plot_shap_analysis(model, X_sample, feature_names):
    """SHAP analysis for model interpretability"""
    try:
        import shap
        
        # Create SHAP explainer
        if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict_proba, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # If binary classification, take values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")

def main():
    """Main execution function"""
    print("="*60)
    print("GRADIENT BOOSTED MODELS FOR TELCO CHURN PREDICTION")
    print("="*60)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    # Initialize GBM models class
    gbm = GBMModels(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train base models
    print("\n" + "-"*40)
    print("TRAINING BASE MODELS")
    print("-"*40)
    
    gbm.train_gradient_boosting()
    gbm.train_xgboost()
    gbm.train_lightgbm()
    gbm.train_adaboost()
    
    # Hyperparameter tuning
    print("\n" + "-"*40)
    print("HYPERPARAMETER TUNING")
    print("-"*40)
    
    xgb_best, xgb_params = gbm.hyperparameter_tuning_xgboost()
    lgb_best, lgb_params = gbm.hyperparameter_tuning_lightgbm()
    
    # Feature importance analysis
    print("\n" + "-"*40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("-"*40)
    
    gbm.plot_feature_importance_comparison()
    
    # Model evaluation plots
    print("\n" + "-"*40)
    print("MODEL EVALUATION PLOTS")
    print("-"*40)
    
    gbm.plot_roc_curves()
    gbm.plot_learning_curves()
    
    # SHAP analysis for best model
    if 'XGBoost_Tuned' in gbm.models:
        print("\nPerforming SHAP analysis...")
        X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        plot_shap_analysis(gbm.models['XGBoost_Tuned'], X_sample, list(X_train.columns))
    
    # Final evaluation
    results_df = gbm.final_evaluation()
    
    # Save results
    print("\n" + "-"*40)
    print("SAVING RESULTS")
    print("-"*40)
    
    output_dir = "/Users/ob/Projects/teleco/results"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/gbm_models_results.csv", index=False)
    print(f"Results saved to {output_dir}/gbm_models_results.csv")

if __name__ == "__main__":
    main()
