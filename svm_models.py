# svm_models.py - Support Vector Machine Models for Telco Churn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc,
                           make_scorer)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from prep import get_preprocessed_data

class SVMModels:
    """Class for training and evaluating SVM models"""
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.models = {}
        self.results = {}
        self.training_times = {}
        
    def train_linear_svm(self, C=1.0, max_iter=1000):
        """Train Linear SVM"""
        print("Training Linear SVM...")
        start_time = time.time()
        
        model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=42,
            dual=False
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['LinearSVM'] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.models['LinearSVM'] = model
        metrics = self.evaluate_model(model, 'LinearSVM', has_proba=False)
        
        return model, metrics
    
    def train_rbf_svm(self, C=1.0, gamma='scale'):
        """Train RBF (Gaussian) Kernel SVM"""
        print("Training RBF SVM...")
        start_time = time.time()
        
        model = SVC(
            kernel='rbf',
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42,
            cache_size=500
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['RBF_SVM'] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.models['RBF_SVM'] = model
        metrics = self.evaluate_model(model, 'RBF_SVM')
        
        return model, metrics
    
    def train_polynomial_svm(self, C=1.0, degree=3):
        """Train Polynomial Kernel SVM"""
        print("Training Polynomial SVM...")
        start_time = time.time()
        
        model = SVC(
            kernel='poly',
            C=C,
            degree=degree,
            gamma='scale',
            probability=True,
            random_state=42,
            cache_size=500
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['Polynomial_SVM'] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.models['Polynomial_SVM'] = model
        metrics = self.evaluate_model(model, 'Polynomial_SVM')
        
        return model, metrics
    
    def train_sigmoid_svm(self, C=1.0):
        """Train Sigmoid Kernel SVM"""
        print("Training Sigmoid SVM...")
        start_time = time.time()
        
        model = SVC(
            kernel='sigmoid',
            C=C,
            gamma='scale',
            probability=True,
            random_state=42,
            cache_size=500
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['Sigmoid_SVM'] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.models['Sigmoid_SVM'] = model
        metrics = self.evaluate_model(model, 'Sigmoid_SVM')
        
        return model, metrics
    
    def train_nu_svm(self, nu=0.5, kernel='rbf'):
        """Train Nu-SVM"""
        print("Training Nu-SVM...")
        start_time = time.time()
        
        model = NuSVC(
            nu=nu,
            kernel=kernel,
            gamma='scale',
            probability=True,
            random_state=42,
            cache_size=500
        )
        
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        self.training_times['Nu_SVM'] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.models['Nu_SVM'] = model
        metrics = self.evaluate_model(model, 'Nu_SVM')
        
        return model, metrics
    
    def hyperparameter_tuning_rbf(self):
        """Hyperparameter tuning for RBF SVM"""
        print("\nPerforming RBF SVM Hyperparameter Tuning...")
        print("This may take several minutes...")
        
        # Use smaller subset for faster tuning
        sample_size = min(2000, len(self.X_train))
        X_sample = self.X_train.sample(n=sample_size, random_state=42)
        y_sample = self.y_train[X_sample.index]
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }
        
        svm = SVC(probability=True, random_state=42, cache_size=500)
        
        grid_search = GridSearchCV(
            svm, param_grid,
            cv=3,  # Reduced CV folds for speed
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_sample, y_sample)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Retrain on full training set with best params
        best_model = SVC(
            **grid_search.best_params_,
            probability=True,
            random_state=42,
            cache_size=500
        )
        best_model.fit(self.X_train, self.y_train)
        
        self.models['RBF_SVM_Tuned'] = best_model
        metrics = self.evaluate_model(best_model, 'RBF_SVM_Tuned')
        
        return best_model, grid_search.best_params_
    
    def hyperparameter_tuning_linear(self):
        """Hyperparameter tuning for Linear SVM"""
        print("\nPerforming Linear SVM Hyperparameter Tuning...")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge']
        }
        
        svm = LinearSVC(random_state=42, max_iter=1000, dual=False)
        
        grid_search = GridSearchCV(
            svm, param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        self.models['Linear_SVM_Tuned'] = best_model
        metrics = self.evaluate_model(best_model, 'Linear_SVM_Tuned', has_proba=False)
        
        return best_model, grid_search.best_params_
    
    def train_svm_with_pca(self, n_components=10):
        """Train SVM with PCA dimensionality reduction"""
        print(f"\nTraining SVM with PCA (n_components={n_components})...")
        
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_components)),
            ('svm', SVC(kernel='rbf', gamma='scale', probability=True, random_state=42))
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        
        self.models['SVM_PCA'] = pipeline
        metrics = self.evaluate_model(pipeline, 'SVM_PCA')
        
        # Analyze PCA components
        pca = pipeline.named_steps['pca']
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}...")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
        
        return pipeline, metrics
    
    def evaluate_model(self, model, model_name, has_proba=True):
        """Evaluate model performance"""
        val_pred = model.predict(self.X_val)
        
        metrics = {
            'accuracy': accuracy_score(self.y_val, val_pred),
            'precision': precision_score(self.y_val, val_pred),
            'recall': recall_score(self.y_val, val_pred),
            'f1': f1_score(self.y_val, val_pred)
        }
        
        if has_proba:
            if hasattr(model, 'predict_proba'):
                val_prob = model.predict_proba(self.X_val)[:, 1]
                metrics['auc'] = roc_auc_score(self.y_val, val_prob)
            elif hasattr(model, 'decision_function'):
                val_scores = model.decision_function(self.X_val)
                metrics['auc'] = roc_auc_score(self.y_val, val_scores)
        
        print(f"\n{model_name} Validation Performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")
        
        self.results[model_name] = metrics
        
        return metrics
    
    def plot_decision_boundaries_2d(self):
        """Visualize decision boundaries in 2D using PCA"""
        print("\nCreating 2D decision boundary visualizations...")
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(self.X_train)
        X_val_2d = pca.transform(self.X_val)
        
        # Train simple models on 2D data for visualization
        models_2d = {
            'Linear': LinearSVC(random_state=42),
            'RBF': SVC(kernel='rbf', gamma='scale', random_state=42),
            'Polynomial': SVC(kernel='poly', degree=3, gamma='scale', random_state=42)
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, model) in enumerate(models_2d.items()):
            # Train on 2D data
            model.fit(X_train_2d, self.y_train)
            
            # Create mesh
            h = 0.02
            x_min, x_max = X_val_2d[:, 0].min() - 1, X_val_2d[:, 0].max() + 1
            y_min, y_max = X_val_2d[:, 1].min() - 1, X_val_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            axes[idx].contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu_r')
            scatter = axes[idx].scatter(X_val_2d[:, 0], X_val_2d[:, 1], 
                                      c=self.y_val, cmap='RdYlBu_r',
                                      edgecolors='black', s=30, alpha=0.6)
            
            axes[idx].set_xlabel('First Principal Component')
            axes[idx].set_ylabel('Second Principal Component')
            axes[idx].set_title(f'{name} SVM Decision Boundary')
        
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/svm_decision_boundaries_2d.png")
        plt.show()
    
    def plot_margin_analysis(self):
        """Analyze and visualize SVM margins"""
        print("\nAnalyzing SVM margins...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        models_to_analyze = ['LinearSVM', 'RBF_SVM', 'Polynomial_SVM', 'Sigmoid_SVM']
        
        for idx, model_name in enumerate(models_to_analyze):
            if model_name in self.models:
                model = self.models[model_name]
                
                # Get decision function values
                if hasattr(model, 'decision_function'):
                    decision_values = model.decision_function(self.X_val)
                    
                    # Plot histogram of decision values
                    axes[idx].hist(decision_values[self.y_val == 0], bins=30, 
                                 alpha=0.5, label='No Churn', color='blue')
                    axes[idx].hist(decision_values[self.y_val == 1], bins=30,
                                 alpha=0.5, label='Churn', color='red')
                    
                    # Add margin lines
                    axes[idx].axvline(x=-1, color='black', linestyle='--', alpha=0.5)
                    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=2)
                    axes[idx].axvline(x=1, color='black', linestyle='--', alpha=0.5)
                    
                    axes[idx].set_xlabel('Decision Function Value')
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].set_title(f'{model_name} - Margin Distribution')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/svm_margin_analysis.png")
        plt.show()
    
    def plot_kernel_comparison(self):
        """Compare different kernel functions"""
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        kernel_results = []
        
        print("\nComparing different kernel functions...")
        
        for kernel in kernels:
            svm = SVC(kernel=kernel, gamma='scale', random_state=42)
            scores = cross_val_score(svm, self.X_train, self.y_train, 
                                    cv=3, scoring='roc_auc', n_jobs=-1)
            
            kernel_results.append({
                'Kernel': kernel,
                'Mean_AUC': scores.mean(),
                'Std_AUC': scores.std()
            })
            
            print(f"{kernel.capitalize()} kernel: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Plot comparison
        results_df = pd.DataFrame(kernel_results)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['Kernel'], results_df['Mean_AUC'])
        plt.errorbar(range(len(results_df)), results_df['Mean_AUC'], 
                    yerr=results_df['Std_AUC'], fmt='none', color='black', capsize=5)
        
        # Color best performer
        best_idx = results_df['Mean_AUC'].idxmax()
        bars[best_idx].set_color('green')
        
        plt.xlabel('Kernel Function')
        plt.ylabel('Mean AUC Score')
        plt.title('SVM Kernel Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/svm_kernel_comparison.png")
        plt.show()
        
        return results_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all SVM models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(self.X_test)
                # Convert to probability-like scores
                y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            else:
                continue
            
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_score = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - SVM Models', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/svm_roc_curves.png")
        plt.show()
    
    def plot_training_time_comparison(self):
        """Plot training time comparison"""
        if self.training_times:
            plt.figure(figsize=(10, 6))
            
            models = list(self.training_times.keys())
            times = list(self.training_times.values())
            
            bars = plt.bar(models, times)
            plt.xlabel('Model')
            plt.ylabel('Training Time (seconds)')
            plt.title('SVM Training Time Comparison')
            plt.xticks(rotation=45, ha='right')
            
            # Color fastest and slowest
            min_idx = times.index(min(times))
            max_idx = times.index(max(times))
            bars[min_idx].set_color('green')
            bars[max_idx].set_color('red')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            output_dir = "data/plots"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/svm_training_times.png")
            plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = min(4, len(self.models))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (model_name, model) in enumerate(list(self.models.items())[:n_models]):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices - SVM Models', fontsize=14)
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/svm_confusion_matrices.png")
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
            
            # Add AUC
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
            
            # Add training time if available
            if model_name in self.training_times:
                metrics['Training_Time'] = self.training_times[model_name]
            
            results.append(metrics)
            
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                if metric not in ['Model', 'Training_Time']:
                    if not np.isnan(value):
                        print(f"  {metric}: {value:.4f}")
            if 'Training_Time' in metrics:
                print(f"  Training Time: {metrics['Training_Time']:.2f} seconds")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AUC', ascending=False, na_position='last')
        
        return results_df
    
    def save_models(self, output_dir=None):
        """Save all trained models"""
        if output_dir is None:
            output_dir = "models"
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f"svm_{model_name.lower()}.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")

def main():
    """Main execution function"""
    print("="*60)
    print("SUPPORT VECTOR MACHINE MODELS FOR TELCO CHURN PREDICTION")
    print("="*60)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    # Initialize SVM models class
    svm_models = SVMModels(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Train different SVM variants
    print("\n" + "-"*40)
    print("TRAINING SVM MODELS")
    print("-"*40)
    
    # 1. Linear SVM
    svm_models.train_linear_svm()
    
    # 2. RBF SVM
    svm_models.train_rbf_svm()
    
    # 3. Polynomial SVM
    svm_models.train_polynomial_svm(degree=3)
    
    # 4. Sigmoid SVM
    svm_models.train_sigmoid_svm()
    
    # 5. Nu-SVM
    svm_models.train_nu_svm(nu=0.5)
    
    # 6. SVM with PCA
    svm_models.train_svm_with_pca(n_components=20)
    
    # 7. Hyperparameter tuning
    print("\n" + "-"*40)
    print("HYPERPARAMETER TUNING")
    print("-"*40)
    
    # Tune RBF SVM
    rbf_best, rbf_params = svm_models.hyperparameter_tuning_rbf()
    
    # Tune Linear SVM
    linear_best, linear_params = svm_models.hyperparameter_tuning_linear()
    
    # 8. Analysis and Visualizations
    print("\n" + "-"*40)
    print("ANALYSIS AND VISUALIZATIONS")
    print("-"*40)
    
    # Kernel comparison
    kernel_results = svm_models.plot_kernel_comparison()
    
    # Decision boundaries
    svm_models.plot_decision_boundaries_2d()
    
    # Margin analysis
    svm_models.plot_margin_analysis()
    
    # ROC curves
    svm_models.plot_roc_curves()
    
    # Confusion matrices
    svm_models.plot_confusion_matrices()
    
    # Training time comparison
    svm_models.plot_training_time_comparison()
    
    # 9. Final evaluation
    results_df = svm_models.final_evaluation()
    
    # 10. Save results and models
    print("\n" + "-"*40)
    print("SAVING RESULTS")
    print("-"*40)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    results_df.to_csv(f"{output_dir}/svm_models_results.csv", index=False)
    print(f"Results saved to {output_dir}/svm_models_results.csv")
    
    # Save kernel comparison results
    kernel_results.to_csv(f"{output_dir}/svm_kernel_comparison.csv", index=False)
    
    # Save models
    svm_models.save_models()
    
    # 11. Summary
    print("\n" + "="*60)
    print("SUMMARY - TOP PERFORMING SVM MODELS")
    print("="*60)
    print(results_df[['Model', 'AUC', 'Accuracy', 'F1']].head(3).to_string(index=False))
    
    print("\n" + "="*60)
    print("SVM MODELS TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()