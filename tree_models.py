# tree_models.py - Decision Tree and Random Forest for Telco Churn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from prep import get_preprocessed_data

def train_decision_tree(X_train, y_train, X_val, y_val, max_depth=None, 
                        min_samples_split=2, min_samples_leaf=1):
    """Train a Decision Tree classifier"""
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # Predictions
    train_pred = dt_model.predict(X_train)
    val_pred = dt_model.predict(X_val)
    train_prob = dt_model.predict_proba(X_train)[:, 1]
    val_prob = dt_model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, train_prob)
    val_metrics = calculate_metrics(y_val, val_pred, val_prob)
    
    print("Decision Tree Performance:")
    print(f"Training - Accuracy: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
    
    return dt_model, train_metrics, val_metrics

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, 
                       max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Train a Random Forest classifier"""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    train_prob = rf_model.predict_proba(X_train)[:, 1]
    val_prob = rf_model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, train_prob)
    val_metrics = calculate_metrics(y_val, val_pred, val_prob)
    
    print("\nRandom Forest Performance:")
    print(f"Training - Accuracy: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
    
    return rf_model, train_metrics, val_metrics

def hyperparameter_tuning_dt(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning for Decision Tree"""
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Best Decision Tree Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
    
    # Evaluate on validation set
    val_pred = best_model.predict(X_val)
    val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, val_pred, val_prob)
    print(f"Validation AUC: {val_metrics['auc']:.4f}")
    
    return best_model, grid_search.best_params_

def hyperparameter_tuning_rf(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning for Random Forest"""
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=5, 
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print("\nBest Random Forest Parameters:", random_search.best_params_)
    print("Best CV Score:", random_search.best_score_)
    
    # Evaluate on validation set
    val_pred = best_model.predict(X_val)
    val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, val_pred, val_prob)
    print(f"Validation AUC: {val_metrics['auc']:.4f}")
    
    return best_model, random_search.best_params_

def calculate_metrics(y_true, y_pred, y_prob=None):
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

def plot_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

def plot_confusion_matrices(models_dict, X_test, y_test):
    """Plot confusion matrices for multiple models"""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrices_trees.png")
    plt.show()

def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Tree-Based Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/roc_curves_trees.png")
    plt.show()

def visualize_decision_tree(model, feature_names, max_depth=3):
    """Visualize decision tree structure"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=['No Churn', 'Churn'],
              filled=True, 
              rounded=True,
              max_depth=max_depth,
              fontsize=9)
    plt.title("Decision Tree Visualization")
    
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/decision_tree_viz.png", dpi=100, bbox_inches='tight')
    plt.show()

def evaluate_final_model(model, X_test, y_test, model_name="Model"):
    """Final evaluation on test set"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"Final Test Set Performance - {model_name}")
    print(f"{'='*50}")
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return metrics

def save_model(model, model_name, output_dir=None):
    """Save trained model"""
    if output_dir is None:
        output_dir = "models"
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def main():
    """Main execution function"""
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    print("\n" + "="*50)
    print("TRAINING TREE-BASED MODELS")
    print("="*50)
    
    # 1. Basic Decision Tree
    print("\n1. Training Basic Decision Tree...")
    dt_model, dt_train_metrics, dt_val_metrics = train_decision_tree(
        X_train, y_train, X_val, y_val, max_depth=5
    )
    
    # 2. Basic Random Forest
    print("\n2. Training Basic Random Forest...")
    rf_model, rf_train_metrics, rf_val_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, n_estimators=100
    )
    
    # 3. Hyperparameter Tuning
    print("\n3. Hyperparameter Tuning...")
    print("\nTuning Decision Tree...")
    best_dt_model, best_dt_params = hyperparameter_tuning_dt(X_train, y_train, X_val, y_val)
    
    print("\nTuning Random Forest...")
    best_rf_model, best_rf_params = hyperparameter_tuning_rf(X_train, y_train, X_val, y_val)
    
    # 4. Feature Importance Analysis
    print("\n4. Analyzing Feature Importance...")
    feature_names = list(X_train.columns)
    
    dt_importance = plot_feature_importance(best_dt_model, feature_names, "Decision Tree", 20)
    rf_importance = plot_feature_importance(best_rf_model, feature_names, "Random Forest", 20)
    
    print("\nTop 10 Most Important Features (Random Forest):")
    print(rf_importance.head(10))
    
    # 5. Visualizations
    print("\n5. Creating Visualizations...")
    
    # Decision tree visualization
    visualize_decision_tree(best_dt_model, feature_names, max_depth=3)
    
    # Confusion matrices
    models_dict = {
        'Decision Tree': best_dt_model,
        'Random Forest': best_rf_model
    }
    plot_confusion_matrices(models_dict, X_test, y_test)
    
    # ROC curves
    plot_roc_curves(models_dict, X_test, y_test)
    
    # 6. Final Evaluation on Test Set
    print("\n6. Final Model Evaluation on Test Set...")
    dt_test_metrics = evaluate_final_model(best_dt_model, X_test, y_test, "Decision Tree")
    rf_test_metrics = evaluate_final_model(best_rf_model, X_test, y_test, "Random Forest")
    
    # 7. Save Models
    print("\n7. Saving Models...")
    save_model(best_dt_model, "decision_tree_best")
    save_model(best_rf_model, "random_forest_best")
    
    # 8. Summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    results_df = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'Val_AUC': [dt_val_metrics['auc'], rf_val_metrics['auc']],
        'Test_AUC': [dt_test_metrics['auc'], rf_test_metrics['auc']],
        'Test_Accuracy': [dt_test_metrics['accuracy'], rf_test_metrics['accuracy']],
        'Test_Precision': [dt_test_metrics['precision'], rf_test_metrics['precision']],
        'Test_Recall': [dt_test_metrics['recall'], rf_test_metrics['recall']],
        'Test_F1': [dt_test_metrics['f1'], rf_test_metrics['f1']]
    })
    
    print(results_df.to_string(index=False))
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/tree_models_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/tree_models_results.csv")

if __name__ == "__main__":
    main()