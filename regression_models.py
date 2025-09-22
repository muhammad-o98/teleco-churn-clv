# regression_models.py - Logistic Regression and Linear Models for Telco Churn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import StandardScaler
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                 RidgeClassifier, SGDClassifier, Perceptron)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from prep import get_preprocessed_data

class RegressionModels:
    """Class for training and evaluating regression-based models"""
    
    # Import StandardScaler
    from sklearn.preprocessing import StandardScaler
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, penalty='l2', C=1.0, max_iter=1000):
        """Train standard Logistic Regression"""
        print("Training Logistic Regression...")
        
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=42,
            solver='liblinear' if penalty == 'l1' else 'lbfgs'
        )
        
        model.fit(self.X_train, self.y_train)
        
        self.models['LogisticRegression'] = model
        metrics = self.evaluate_model(model, 'LogisticRegression')
        
        # Store coefficients for interpretation
        self.analyze_coefficients(model, 'LogisticRegression')
        
        return model, metrics
    
    def train_logistic_regression_cv(self):
        """Train Logistic Regression with built-in cross-validation"""
        print("Training Logistic Regression with CV...")
        
        model = LogisticRegressionCV(
            cv=5,
            penalty='l2',
            max_iter=1000,
            random_state=42,
            scoring='roc_auc'
        )
        
        model.fit(self.X_train, self.y_train)
        
        print(f"Best C value from CV: {model.C_[0]:.4f}")
        
        self.models['LogisticRegressionCV'] = model
        metrics = self.evaluate_model(model, 'LogisticRegressionCV')
        
        return model, metrics
    
    def train_ridge_classifier(self, alpha=1.0):
        """Train Ridge Classifier"""
        print("Training Ridge Classifier...")
        
        model = RidgeClassifier(
            alpha=alpha,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        self.models['RidgeClassifier'] = model
        metrics = self.evaluate_model(model, 'RidgeClassifier', has_proba=False)
        
        return model, metrics
    
    def train_elastic_net(self, alpha=0.1, l1_ratio=0.5):
        """Train Elastic Net (via SGD)"""
        print("Training Elastic Net...")
        
        model = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        self.models['ElasticNet'] = model
        metrics = self.evaluate_model(model, 'ElasticNet', has_proba=True)
        
        return model, metrics
    
    def train_polynomial_logistic(self, degree=2):
        """Train Logistic Regression with polynomial features"""
        print(f"Training Polynomial Logistic Regression (degree={degree})...")
        
        # Create pipeline with polynomial features
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(max_iter=1000, random_state=42))
        ])
        

        
        pipeline.fit(self.X_train, self.y_train)
        
        self.models['PolynomialLogistic'] = pipeline
        metrics = self.evaluate_model(pipeline, 'PolynomialLogistic')
        
        return pipeline, metrics
    
    def hyperparameter_tuning(self):
        """Comprehensive hyperparameter tuning for Logistic Regression"""
        print("\nPerforming Hyperparameter Tuning...")
        
        # Parameter grid for different penalties
        param_grids = [
            {
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            },
            {
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
            },
            {
                'penalty': ['elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['saga'],
                'l1_ratio': [0.2, 0.5, 0.8]
            }
        ]
        
        best_model = None
        best_score = -1
        best_params = None
        
        for param_grid in param_grids:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            
            grid_search = GridSearchCV(
                lr, param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
        print(f"Best Parameters: {best_params}")
        print(f"Best CV AUC Score: {best_score:.4f}")
        
        self.models['LogisticRegression_Tuned'] = best_model
        metrics = self.evaluate_model(best_model, 'LogisticRegression_Tuned')
        
        return best_model, best_params
    
    def evaluate_model(self, model, model_name, has_proba=True):
        """Evaluate model performance"""
        # Predictions
        val_pred = model.predict(self.X_val)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, val_pred),
            'precision': precision_score(self.y_val, val_pred),
            'recall': recall_score(self.y_val, val_pred),
            'f1': f1_score(self.y_val, val_pred)
        }
        
        if has_proba:
            if hasattr(model, 'predict_proba'):
                val_prob = model.predict_proba(self.X_val)[:, 1]
            elif hasattr(model, '_predict_proba_lr'):
                val_prob = model._predict_proba_lr(self.X_val)[:, 1]
            else:
                val_prob = model.decision_function(self.X_val)
                # Convert decision function to probability-like scores
                val_prob = 1 / (1 + np.exp(-val_prob))
            
            metrics['auc'] = roc_auc_score(self.y_val, val_prob)
        
        print(f"\n{model_name} Validation Performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")
        
        self.results[model_name] = metrics
        
        return metrics
    
    def analyze_coefficients(self, model, model_name):
        """Analyze and visualize logistic regression coefficients"""
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            feature_names = self.X_train.columns
            
            # Create DataFrame of coefficients
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            # Plot top coefficients
            plt.figure(figsize=(10, 8))
            top_n = 20
            top_features = coef_df.head(top_n)
            
            colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
            plt.barh(range(top_n), top_features['Coefficient'], color=colors)
            plt.yticks(range(top_n), top_features['Feature'])
            plt.xlabel('Coefficient Value')
            plt.title(f'Top {top_n} Feature Coefficients - {model_name}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add legend
            red_patch = plt.Rectangle((0,0),1,1, color='red', label='Negative (decreases churn)')
            blue_patch = plt.Rectangle((0,0),1,1, color='blue', label='Positive (increases churn)')
            plt.legend(handles=[red_patch, blue_patch], loc='lower right')
            
            plt.tight_layout()
            
            output_dir = "data/plots"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/logistic_coefficients_{model_name}.png")
            plt.show()
            
            return coef_df
    
    def plot_calibration_curve(self):
        """Plot calibration curves for probabilistic models"""
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Calibration plots
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_val)[:, 1]
                
                fraction_pos, mean_pred = calibration_curve(
                    self.y_val, y_prob, n_bins=10, strategy='uniform'
                )
                
                ax1.plot(mean_pred, fraction_pos, 'o-', label=model_name)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curves')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of predictions
        ax2 = axes[1]
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_val)[:, 1]
                ax2.hist(y_prob, bins=30, alpha=0.3, label=model_name)
        
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/calibration_curves.png")
        plt.show()
    
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(self.X_test)
            else:
                continue
            
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            avg_precision = average_precision_score(self.y_test, y_prob)
            
            plt.plot(recall, precision, 
                    label=f'{model_name} (AP = {avg_precision:.3f})',
                    linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Regression Models', fontsize=14)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/precision_recall_curves_regression.png")
        plt.show()
    
    def plot_decision_boundary(self, model, model_name, features_idx=[0, 1]):
        """Plot decision boundary for 2D visualization"""
        # Select two features for visualization
        X_subset = self.X_val.iloc[:, features_idx].values
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create full feature array with zeros for other features
        mesh_data = np.zeros((xx.ravel().shape[0], self.X_val.shape[1]))
        mesh_data[:, features_idx[0]] = xx.ravel()
        mesh_data[:, features_idx[1]] = yy.ravel()
        
        # Predict on mesh
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(mesh_data)[:, 1]
        else:
            Z = model.decision_function(mesh_data)
            Z = 1 / (1 + np.exp(-Z))  # Convert to probability
        
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu_r')
        plt.colorbar(label='Churn Probability')
        
        # Plot actual points
        scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], 
                            c=self.y_val, cmap='RdYlBu_r', 
                            edgecolors='black', s=50, alpha=0.6)
        
        plt.xlabel(self.X_val.columns[features_idx[0]])
        plt.ylabel(self.X_val.columns[features_idx[1]])
        plt.title(f'Decision Boundary - {model_name}')
        plt.legend(*scatter.legend_elements(), title="Actual Churn")
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/decision_boundary_{model_name}.png")
        plt.show()
    
    def cross_validation_analysis(self):
        """Perform detailed cross-validation analysis"""
        print("\n" + "="*50)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*50)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name not in ['PolynomialLogistic']:  # Skip pipeline models
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=5, scoring='roc_auc', n_jobs=-1
                )
                
                cv_results[model_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                
                print(f"\n{model_name}:")
                print(f"  Mean AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                print(f"  Scores: {scores}")
        
        return cv_results
    
    def plot_learning_curves(self):
        """Plot learning curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        model_names = list(self.models.keys())[:4]
        
        for idx, model_name in enumerate(model_names):
            model = self.models[model_name]
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train,
                cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='roc_auc'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[idx].plot(train_sizes, train_mean, 'o-', label='Training')
            axes[idx].fill_between(train_sizes, train_mean - train_std,
                                  train_mean + train_std, alpha=0.1)
            
            axes[idx].plot(train_sizes, val_mean, 'o-', label='Validation')
            axes[idx].fill_between(train_sizes, val_mean - val_std,
                                  val_mean + val_std, alpha=0.1)
            
            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('AUC Score')
            axes[idx].set_title(f'{model_name} Learning Curve')
            axes[idx].legend(loc='lower right')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/learning_curves_regression.png")
        plt.show()
    
    def final_evaluation(self):
        """Final evaluation on test set"""
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred)
            }
            
            # Add AUC if model supports probability predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(self.X_test)
                try:
                    metrics['AUC'] = roc_auc_score(self.y_test, y_scores)
                except:
                    metrics['AUC'] = np.nan
            else:
                metrics['AUC'] = np.nan
            
            results.append(metrics)
            
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                if metric != 'Model' and not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AUC', ascending=False, na_position='last')
        
        return results_df
    
    def save_models(self, output_dir=None):
        """Save all trained models"""
        if output_dir is None:
            output_dir = "models"
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f"regression_{model_name.lower()}.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")

def main():
    """Main execution function"""
    print("="*60)
    print("REGRESSION MODELS FOR TELCO CHURN PREDICTION")
    print("="*60)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    # Initialize regression models class
    reg_models = RegressionModels(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train different regression models
    print("\n" + "-"*40)
    print("TRAINING REGRESSION MODELS")
    print("-"*40)
    
    # 1. Standard Logistic Regression
    reg_models.train_logistic_regression()
    
    # 2. L1 Regularized Logistic Regression
    reg_models.train_logistic_regression(penalty='l1', C=0.1)
    reg_models.models['LogisticRegression_L1'] = reg_models.models.pop('LogisticRegression')
    
    # 3. Logistic Regression with CV
    reg_models.train_logistic_regression_cv()
    
    # 4. Ridge Classifier
    reg_models.train_ridge_classifier()
    
    # 5. Elastic Net
    reg_models.train_elastic_net()
    
    # 6. Polynomial Logistic Regression
    reg_models.train_polynomial_logistic(degree=2)
    
    # 7. Hyperparameter tuning
    print("\n" + "-"*40)
    print("HYPERPARAMETER TUNING")
    print("-"*40)
    
    best_model, best_params = reg_models.hyperparameter_tuning()
    
    # 8. Cross-validation analysis
    cv_results = reg_models.cross_validation_analysis()
    
    # 9. Visualizations
    print("\n" + "-"*40)
    print("CREATING VISUALIZATIONS")
    print("-"*40)
    
    # Calibration curves
    reg_models.plot_calibration_curve()
    
    # Precision-Recall curves
    reg_models.plot_precision_recall_curves()
    
    # Learning curves
    reg_models.plot_learning_curves()
    
    # Decision boundary for best model
    if 'LogisticRegression_Tuned' in reg_models.models:
        reg_models.plot_decision_boundary(
            reg_models.models['LogisticRegression_Tuned'],
            'LogisticRegression_Tuned',
            features_idx=[0, 1]  # Using first two features
        )
    
    # 10. Final evaluation
    results_df = reg_models.final_evaluation()
    
    # 11. Save results and models
    print("\n" + "-"*40)
    print("SAVING RESULTS")
    print("-"*40)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/regression_models_results.csv", index=False)
    print(f"Results saved to {output_dir}/regression_models_results.csv")
    
    reg_models.save_models()
    
    # 12. Summary
    print("\n" + "="*60)
    print("SUMMARY - TOP PERFORMING MODELS")
    print("="*60)
    print(results_df.head(3).to_string(index=False))
    
    print("\n" + "="*60)
    print("REGRESSION MODELS TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()