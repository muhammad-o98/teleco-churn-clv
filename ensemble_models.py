# ensemble_models.py - Model Comparison and Ensemble Methods for Telco Churn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc,
                           precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from prep import get_preprocessed_data

# Import models from other modules (these would be imported if modules exist)
# from tree_models import train_decision_tree, train_random_forest
# from gbm_models import GBMModels
# from regression_models import RegressionModels
# from svm_models import SVMModels

class ModelComparison:
    """Class for comprehensive model comparison and ensemble methods"""
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        self.ensemble_models = {}
        
    def load_trained_models(self, models_dir="/Users/ob/Projects/teleco/models"):
        """Load pre-trained models from disk"""
        print("Loading pre-trained models...")
        
        model_files = [
            'decision_tree_best.pkl',
            'random_forest_best.pkl',
            'gbm_xgboost_tuned.pkl',
            'gbm_lightgbm_tuned.pkl',
            'regression_logisticregression_tuned.pkl',
            'svm_rbf_svm_tuned.pkl'
        ]
        
        for file in model_files:
            filepath = os.path.join(models_dir, file)
            if os.path.exists(filepath):
                model_name = file.replace('.pkl', '').replace('_', ' ').title()
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded: {model_name}")
        
        if not self.models:
            print("No pre-trained models found. Training basic models...")
            self.train_basic_models()
    
    def train_basic_models(self):
        """Train basic versions of each model type for comparison"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import xgboost as xgb
        import lightgbm as lgb
        
        print("\nTraining basic models for comparison...")
        
        models_config = [
            ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('XGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('LightGBM', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
            ('SVM', SVC(kernel='rbf', probability=True, random_state=42))
        ]
        
        for name, model in models_config:
            print(f"Training {name}...")
            start_time = time.time()
            
            model.fit(self.X_train, self.y_train)
            
            training_time = time.time() - start_time
            self.models[name] = model
            
            # Generate predictions
            self.predictions[name] = model.predict(self.X_test)
            if hasattr(model, 'predict_proba'):
                self.probabilities[name] = model.predict_proba(self.X_test)[:, 1]
            
            print(f"  Completed in {training_time:.2f} seconds")
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            self.predictions[name] = y_pred
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred)
            }
            
            # Add AUC if model supports probabilities
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                self.probabilities[name] = y_prob
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)
            
            self.metrics[name] = metrics
            results.append(metrics)
            
            print(f"\n{name}:")
            for metric, value in metrics.items():
                if metric != 'Model':
                    print(f"  {metric}: {value:.4f}")
        
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('AUC', ascending=False, na_position='last')
        
        return self.results_df
    
    def create_voting_ensemble(self, voting='soft'):
        """Create voting ensemble"""
        print(f"\nCreating Voting Ensemble ({voting} voting)...")
        
        # Select best models based on AUC
        top_models = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                top_models.append((name, model))
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=top_models[:5],  # Use top 5 models
            voting=voting
        )
        
        # Train ensemble
        voting_clf.fit(self.X_train, self.y_train)
        
        self.ensemble_models['Voting Ensemble'] = voting_clf
        
        # Evaluate
        y_pred = voting_clf.predict(self.X_test)
        y_prob = voting_clf.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'Model': 'Voting Ensemble',
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, y_prob)
        }
        
        print("Voting Ensemble Performance:")
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"  {metric}: {value:.4f}")
        
        return voting_clf, metrics
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble"""
        print("\nCreating Stacking Ensemble...")
        
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        base_models = []
        for name, model in list(self.models.items())[:5]:
            base_models.append((name, model))
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5  # Use cross-validation to train meta-learner
        )
        
        # Train ensemble
        stacking_clf.fit(self.X_train, self.y_train)
        
        self.ensemble_models['Stacking Ensemble'] = stacking_clf
        
        # Evaluate
        y_pred = stacking_clf.predict(self.X_test)
        y_prob = stacking_clf.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'Model': 'Stacking Ensemble',
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, y_prob)
        }
        
        print("Stacking Ensemble Performance:")
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"  {metric}: {value:.4f}")
        
        return stacking_clf, metrics
    
    def create_weighted_average_ensemble(self):
        """Create weighted average ensemble based on model performance"""
        print("\nCreating Weighted Average Ensemble...")
        
        # Calculate weights based on AUC scores
        weights = {}
        total_auc = 0
        
        for name in self.probabilities.keys():
            if name in self.metrics:
                auc_score = self.metrics[name].get('AUC', 0)
                weights[name] = auc_score
                total_auc += auc_score
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_auc
        
        print("Model weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Create weighted predictions
        weighted_probs = np.zeros(len(self.X_test))
        for name, weight in weights.items():
            weighted_probs += self.probabilities[name] * weight
        
        # Convert to binary predictions
        y_pred = (weighted_probs >= 0.5).astype(int)
        
        metrics = {
            'Model': 'Weighted Average Ensemble',
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, weighted_probs)
        }
        
        print("\nWeighted Average Ensemble Performance:")
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"  {metric}: {value:.4f}")
        
        self.ensemble_models['Weighted Average'] = {'weights': weights, 'metrics': metrics}
        
        return weights, metrics
    
    def plot_comprehensive_comparison(self):
        """Create comprehensive comparison plots"""
        # Combine all results
        all_results = self.results_df.copy()
        
        # Add ensemble results
        for name, model in self.ensemble_models.items():
            if isinstance(model, dict) and 'metrics' in model:
                all_results = pd.concat([all_results, pd.DataFrame([model['metrics']])], 
                                       ignore_index=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. AUC Comparison
        ax1 = axes[0, 0]
        data = all_results.sort_values('AUC', ascending=True)
        bars = ax1.barh(range(len(data)), data['AUC'])
        ax1.set_yticks(range(len(data)))
        ax1.set_yticklabels(data['Model'])
        ax1.set_xlabel('AUC Score')
        ax1.set_title('Model Comparison - AUC')
        ax1.grid(True, alpha=0.3)
        
        # Color best performer
        bars[-1].set_color('green')
        bars[-2].set_color('lightgreen')
        
        # 2. Accuracy vs Recall Trade-off
        ax2 = axes[0, 1]
        ax2.scatter(all_results['Recall'], all_results['Accuracy'], s=100, alpha=0.6)
        for idx, row in all_results.iterrows():
            ax2.annotate(row['Model'][:10], (row['Recall'], row['Accuracy']), 
                        fontsize=8, rotation=15)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Recall Trade-off')
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision vs Recall
        ax3 = axes[0, 2]
        ax3.scatter(all_results['Recall'], all_results['Precision'], s=100, alpha=0.6)
        for idx, row in all_results.iterrows():
            ax3.annotate(row['Model'][:10], (row['Recall'], row['Precision']), 
                        fontsize=8, rotation=15)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # 4. F1 Score Comparison
        ax4 = axes[1, 0]
        data = all_results.sort_values('F1', ascending=True)
        bars = ax4.barh(range(len(data)), data['F1'])
        ax4.set_yticks(range(len(data)))
        ax4.set_yticklabels(data['Model'])
        ax4.set_xlabel('F1 Score')
        ax4.set_title('Model Comparison - F1 Score')
        ax4.grid(True, alpha=0.3)
        
        # 5. Radar Chart for Top 5 Models
        ax5 = axes[1, 1]
        top_models = all_results.nlargest(5, 'AUC')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        for idx, row in top_models.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]
            ax5.plot(angles, values, 'o-', linewidth=2, label=row['Model'][:15])
            ax5.fill(angles, values, alpha=0.1)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics)
        ax5.set_ylim(0, 1)
        ax5.set_title('Top 5 Models - Radar Chart')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True)
        
        # 6. Ensemble vs Individual Performance
        ax6 = axes[1, 2]
        ensemble_mask = all_results['Model'].str.contains('Ensemble', na=False)
        
        colors = ['red' if ensemble else 'blue' 
                 for ensemble in ensemble_mask]
        ax6.scatter(all_results['AUC'], all_results['F1'], 
                   c=colors, s=100, alpha=0.6)
        
        for idx, row in all_results.iterrows():
            ax6.annotate(row['Model'][:10], (row['AUC'], row['F1']), 
                        fontsize=8, rotation=15)
        
        ax6.set_xlabel('AUC')
        ax6.set_ylabel('F1 Score')
        ax6.set_title('Ensemble vs Individual Models')
        ax6.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Individual'),
                          Patch(facecolor='red', label='Ensemble')]
        ax6.legend(handles=legend_elements)
        
        plt.suptitle('Comprehensive Model Comparison', fontsize=16)
        plt.tight_layout()
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/comprehensive_model_comparison.png", dpi=100, bbox_inches='tight')
        plt.show()
        
        return all_results
    
    def plot_roc_curves_all(self):
        """Plot ROC curves for all models including ensembles"""
        plt.figure(figsize=(12, 8))
        
        # Plot ROC for each model
        for name, probs in self.probabilities.items():
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Add weighted ensemble if exists
        if 'Weighted Average' in self.ensemble_models:
            weights_dict = self.ensemble_models['Weighted Average']['weights']
            weighted_probs = np.zeros(len(self.X_test))
            for model_name, weight in weights_dict.items():
                if model_name in self.probabilities:
                    weighted_probs += self.probabilities[model_name] * weight
            
            fpr, tpr, _ = roc_curve(self.y_test, weighted_probs)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Weighted Ensemble (AUC = {auc_score:.3f})', 
                    linewidth=3, linestyle='--', color='red')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        plt.savefig(f"{output_dir}/roc_curves_all_models.png")
        plt.show()
    
    def analyze_prediction_agreement(self):
        """Analyze agreement between different models"""
        print("\n" + "="*50)
        print("ANALYZING PREDICTION AGREEMENT")
        print("="*50)
        
        # Create DataFrame of predictions
        pred_df = pd.DataFrame(self.predictions)
        
        # Calculate pairwise agreement
        n_models = len(pred_df.columns)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i, col1 in enumerate(pred_df.columns):
            for j, col2 in enumerate(pred_df.columns):
                agreement = (pred_df[col1] == pred_df[col2]).mean()
                agreement_matrix[i, j] = agreement
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, 
                   xticklabels=pred_df.columns,
                   yticklabels=pred_df.columns,
                   annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Agreement Rate'})
        plt.title('Model Prediction Agreement Matrix')
        plt.tight_layout()
        
        output_dir = "/Users/ob/Projects/teleco/data/plots"
        plt.savefig(f"{output_dir}/prediction_agreement.png")
        plt.show()
        
        # Find cases where models disagree
        disagreement = pred_df.std(axis=1)
        high_disagreement_idx = disagreement[disagreement > 0.3].index
        
        print(f"\nNumber of samples with high disagreement: {len(high_disagreement_idx)}")
        print(f"Percentage of total: {len(high_disagreement_idx)/len(pred_df)*100:.2f}%")
        
        return agreement_matrix, high_disagreement_idx
    
    def calibrate_models(self):
        """Calibrate model probabilities"""
        print("\n" + "="*50)
        print("CALIBRATING MODELS")
        print("="*50)
        
        calibrated_models = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                print(f"Calibrating {name}...")
                
                # Use isotonic calibration
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                calibrated.fit(self.X_val, self.y_val)
                
                calibrated_models[f"{name}_Calibrated"] = calibrated
                
                # Evaluate improvement
                original_prob = model.predict_proba(self.X_test)[:, 1]
                calibrated_prob = calibrated.predict_proba(self.X_test)[:, 1]
                
                original_auc = roc_auc_score(self.y_test, original_prob)
                calibrated_auc = roc_auc_score(self.y_test, calibrated_prob)
                
                print(f"  Original AUC: {original_auc:.4f}")
                print(f"  Calibrated AUC: {calibrated_auc:.4f}")
                print(f"  Improvement: {calibrated_auc - original_auc:.4f}")
        
        return calibrated_models
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("FINAL COMPREHENSIVE REPORT")
        print("="*60)
        
        # Best individual model
        best_individual = self.results_df.iloc[0]
        print(f"\nBest Individual Model: {best_individual['Model']}")
        print(f"  AUC: {best_individual['AUC']:.4f}")
        print(f"  F1 Score: {best_individual['F1']:.4f}")
        
        # Best ensemble
        ensemble_results = []
        for name, model in self.ensemble_models.items():
            if isinstance(model, dict) and 'metrics' in model:
                ensemble_results.append(model['metrics'])
        
        if ensemble_results:
            ensemble_df = pd.DataFrame(ensemble_results)
            best_ensemble = ensemble_df.loc[ensemble_df['AUC'].idxmax()]
            print(f"\nBest Ensemble Model: {best_ensemble['Model']}")
            print(f"  AUC: {best_ensemble['AUC']:.4f}")
            print(f"  F1 Score: {best_ensemble['F1']:.4f}")
        
        # Top 5 models overall
        print("\nTop 5 Models Overall:")
        print(self.results_df[['Model', 'AUC', 'F1', 'Accuracy']].head(5).to_string(index=False))
        
        # Business recommendations
        print("\n" + "-"*40)
        print("BUSINESS RECOMMENDATIONS")
        print("-"*40)
        
        print("""
        Based on the comprehensive analysis:
        
        1. Model Deployment:
           - Primary: Deploy the best ensemble model for production
           - Backup: Use top individual model for real-time predictions
           
        2. Threshold Optimization:
           - Current threshold: 0.5
           - Consider adjusting based on business cost of false positives vs false negatives
           
        3. Feature Importance:
           - Focus retention efforts on high-impact features identified
           - Monitor changes in feature distributions
           
        4. Continuous Improvement:
           - Retrain models monthly with new data
           - Monitor model drift and performance degradation
           - A/B test different model versions
        """)
        
        # Save final report
        output_dir = "/Users/ob/Projects/teleco/results"
        
        with open(f"{output_dir}/final_report.txt", 'w') as f:
            f.write("TELCO CHURN PREDICTION - FINAL REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Model: {best_individual['Model']}\n")
            f.write(f"AUC Score: {best_individual['AUC']:.4f}\n")
            f.write(f"F1 Score: {best_individual['F1']:.4f}\n\n")
            f.write("All Models Performance:\n")
            f.write(self.results_df.to_string(index=False))
        
        print(f"\nFinal report saved to {output_dir}/final_report.txt")

def main():
    """Main execution function"""
    print("="*60)
    print("MODEL COMPARISON AND ENSEMBLE ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    # Initialize comparison class
    comparison = ModelComparison(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train or load models
    print("\n" + "-"*40)
    print("LOADING/TRAINING MODELS")
    print("-"*40)
    comparison.train_basic_models()  # Or use load_trained_models() if models exist
    
    # Evaluate all models
    print("\n" + "-"*40)
    print("MODEL EVALUATION")
    print("-"*40)
    results_df = comparison.evaluate_all_models()
    
    # Create ensembles
    print("\n" + "-"*40)
    print("CREATING ENSEMBLE MODELS")
    print("-"*40)
    
    voting_model, voting_metrics = comparison.create_voting_ensemble()
    stacking_model, stacking_metrics = comparison.create_stacking_ensemble()
    weights, weighted_metrics = comparison.create_weighted_average_ensemble()
    
    # Calibration
    print("\n" + "-"*40)
    print("MODEL CALIBRATION")
    print("-"*40)
    calibrated_models = comparison.calibrate_models()
    
    # Analysis
    print("\n" + "-"*40)
    print("COMPREHENSIVE ANALYSIS")
    print("-"*40)
    
    # Prediction agreement
    agreement_matrix, disagreement_idx = comparison.analyze_prediction_agreement()
    
    # Visualizations
    print("\n" + "-"*40)
    print("CREATING VISUALIZATIONS")
    print("-"*40)
    
    all_results = comparison.plot_comprehensive_comparison()
    comparison.plot_roc_curves_all()
    
    # Generate final report
    comparison.generate_final_report()
    
    # Save all results
    print("\n" + "-"*40)
    print("SAVING RESULTS")
    print("-"*40)
    
    output_dir = "/Users/ob/Projects/teleco/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model comparison
    all_results.to_csv(f"{output_dir}/all_models_comparison.csv", index=False)
    print(f"Results saved to {output_dir}/all_models_comparison.csv")
    
    # Save best models
    best_model_name = results_df.iloc[0]['Model']
    if best_model_name in comparison.models:
        best_model = comparison.models[best_model_name]
        joblib.dump(best_model, f"{output_dir}/best_model_final.pkl")
        print(f"Best model saved to {output_dir}/best_model_final.pkl")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()