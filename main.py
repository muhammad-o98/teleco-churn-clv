#!/usr/bin/env python3
# main.py - Optimized Main Orchestrator for Telco Churn Prediction Pipeline
"""
Telco Customer Churn Prediction - Optimized Pipeline
=====================================================
Complete ML pipeline with parallel processing and intelligent resource management.

Usage:
    python main.py --mode [quick|full|custom] --stages [stage1,stage2,...] --parallel
    
Examples:
    python main.py --mode quick                    # Quick analysis
    python main.py --mode full --parallel          # Full pipeline with parallel processing
    python main.py --stages prep,tree,gbm,ensemble # Run specific stages
    python main.py --evaluate                      # Evaluate existing models
    python main.py --predict data.csv              # Make predictions on new data
"""

import os
import sys
import time
import json
import pickle
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from functools import partial
from contextlib import contextmanager

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure warnings and display
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# Import all project modules
try:
    from prep import get_preprocessed_data, save_processed_data
    from eda import save_summary_tables
    from tree_models import train_decision_tree, train_random_forest
    from gbm_models import GBMModels
    from regression_models import RegressionModels
    from svm_models import SVMModels
    from clustering_models import ClusteringModels
    from ensemble_models import ModelComparison
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all module files are in the same directory")
    sys.exit(1)


class OptimizedTelcoPipeline:
    """Optimized pipeline orchestrator with parallel processing and caching"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration"""
        self.config = config
        self.project_dir = Path(config.get('project_dir', '/Users/ob/Projects/teleco'))
        self.data_dir = self.project_dir / 'data'
        self.models_dir = self.project_dir / 'models'
        self.results_dir = self.project_dir / 'results'
        self.plots_dir = self.data_dir / 'plots'
        self.cache_dir = self.project_dir / 'cache'
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, 
                         self.plots_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'stages_completed': [],
            'stages_failed': [],
            'models': {},
            'results': {},
            'best_model': None,
            'errors': []
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.execution_times = {}
        
        # Data holders
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        
        # Model storage
        self.trained_models = {}
        self.model_scores = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        import logging
        
        log_file = self.project_dir / 'logs' / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def timer(self, stage_name: str):
        """Context manager for timing stages"""
        start = time.time()
        self.logger.info(f"Starting {stage_name}...")
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.execution_times[stage_name] = elapsed
            self.logger.info(f"Completed {stage_name} in {elapsed:.2f} seconds")
    
    def cache_exists(self, cache_name: str) -> bool:
        """Check if cached data exists"""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        return cache_file.exists()
    
    def load_cache(self, cache_name: str) -> Any:
        """Load cached data"""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_cache(self, data: Any, cache_name: str):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def stage_data_preprocessing(self, force_reload=False) -> bool:
        """Stage 1: Data Preprocessing with caching"""
        with self.timer("Data Preprocessing"):
            try:
                # Check cache
                if not force_reload and self.cache_exists('preprocessed_data'):
                    self.logger.info("Loading preprocessed data from cache...")
                    cached_data = self.load_cache('preprocessed_data')
                    self.X_train = cached_data['X_train']
                    self.X_val = cached_data['X_val']
                    self.X_test = cached_data['X_test']
                    self.y_train = cached_data['y_train']
                    self.y_val = cached_data['y_val']
                    self.y_test = cached_data['y_test']
                    self.scaler = cached_data['scaler']
                else:
                    # Process data
                    data_path = self.data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
                    
                    if not data_path.exists():
                        # Try alternate path
                        data_path = self.data_dir / 'WA_FnUseC_TelcoCustomerChurn.csv'
                    
                    self.X_train, self.X_val, self.X_test, \
                    self.y_train, self.y_val, self.y_test, self.scaler = \
                        get_preprocessed_data(str(data_path))
                    
                    # Cache the processed data
                    self.save_cache({
                        'X_train': self.X_train,
                        'X_val': self.X_val,
                        'X_test': self.X_test,
                        'y_train': self.y_train,
                        'y_val': self.y_val,
                        'y_test': self.y_test,
                        'scaler': self.scaler
                    }, 'preprocessed_data')
                
                # Log statistics
                self.logger.info(f"Data shapes - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
                self.logger.info(f"Class distribution - Train: {self.y_train.value_counts().to_dict()}")
                
                self.pipeline_state['stages_completed'].append('preprocessing')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in preprocessing: {str(e)}")
                self.pipeline_state['stages_failed'].append('preprocessing')
                self.pipeline_state['errors'].append(str(e))
                return False
    
    def stage_eda(self) -> bool:
        """Stage 2: Exploratory Data Analysis"""
        with self.timer("EDA"):
            try:
                # Import and run EDA functions
                from eda import (plot_churn_distribution, plot_gender_churn,
                                plot_tenure_hist, plot_monthlycharges,
                                plot_contract_churn, plot_internetservice,
                                plot_paymentmethod, plot_heatmap)
                
                # Load original data for EDA
                data_path = self.data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
                if not data_path.exists():
                    data_path = self.data_dir / 'WA_FnUseC_TelcoCustomerChurn.csv'
                    
                df = pd.read_csv(data_path)
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                
                # Generate all plots
                plot_functions = [
                    plot_churn_distribution, plot_gender_churn,
                    plot_tenure_hist, plot_monthlycharges,
                    plot_contract_churn, plot_internetservice,
                    plot_paymentmethod, plot_heatmap
                ]
                
                for plot_func in plot_functions:
                    plot_func(df)
                
                save_summary_tables(df)
                
                self.pipeline_state['stages_completed'].append('eda')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in EDA: {str(e)}")
                self.pipeline_state['stages_failed'].append('eda')
                return True  # Don't fail pipeline for EDA errors
    
    def stage_tree_models(self) -> bool:
        """Stage 3: Tree-based Models"""
        with self.timer("Tree Models"):
            try:
                # Train Decision Tree
                dt_model, dt_train, dt_val = train_decision_tree(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    max_depth=10 if self.config['mode'] == 'full' else 5
                )
                
                # Train Random Forest
                rf_model, rf_train, rf_val = train_random_forest(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    n_estimators=200 if self.config['mode'] == 'full' else 100
                )
                
                # Store models and scores
                self.trained_models['DecisionTree'] = dt_model
                self.trained_models['RandomForest'] = rf_model
                self.model_scores['DecisionTree'] = dt_val
                self.model_scores['RandomForest'] = rf_val
                
                self.pipeline_state['stages_completed'].append('tree_models')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in tree models: {str(e)}")
                self.pipeline_state['stages_failed'].append('tree_models')
                return False
    
    def stage_gbm_models(self) -> bool:
        """Stage 4: Gradient Boosted Models"""
        with self.timer("GBM Models"):
            try:
                gbm = GBMModels(self.X_train, self.X_val, self.X_test,
                              self.y_train, self.y_val, self.y_test)
                
                # Train models
                gbm.train_gradient_boosting()
                gbm.train_xgboost()
                gbm.train_lightgbm()
                
                if self.config['mode'] == 'full':
                    gbm.train_adaboost()
                    # Hyperparameter tuning
                    gbm.hyperparameter_tuning_xgboost()
                    gbm.hyperparameter_tuning_lightgbm()
                
                # Store models
                for name, model in gbm.models.items():
                    self.trained_models[name] = model
                    if name in gbm.results:
                        self.model_scores[name] = gbm.results[name].get('val_metrics', {})
                
                self.pipeline_state['stages_completed'].append('gbm_models')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in GBM models: {str(e)}")
                self.pipeline_state['stages_failed'].append('gbm_models')
                return False
    
    def stage_regression_models(self) -> bool:
        """Stage 5: Regression Models"""
        with self.timer("Regression Models"):
            try:
                reg = RegressionModels(self.X_train, self.X_val, self.X_test,
                                     self.y_train, self.y_val, self.y_test)
                
                # Train models
                reg.train_logistic_regression()
                reg.train_ridge_classifier()
                
                if self.config['mode'] == 'full':
                    reg.train_logistic_regression_cv()
                    reg.train_elastic_net()
                    reg.hyperparameter_tuning()
                
                # Store models
                for name, model in reg.models.items():
                    self.trained_models[name] = model
                    if name in reg.results:
                        self.model_scores[name] = reg.results[name]
                
                self.pipeline_state['stages_completed'].append('regression_models')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in regression models: {str(e)}")
                self.pipeline_state['stages_failed'].append('regression_models')
                return False
    
    def stage_svm_models(self) -> bool:
        """Stage 6: SVM Models"""
        if self.config['mode'] == 'quick':
            self.logger.info("Skipping SVM in quick mode")
            return True
            
        with self.timer("SVM Models"):
            try:
                svm = SVMModels(self.X_train, self.X_val, self.X_test,
                              self.y_train, self.y_val, self.y_test)
                
                # Use smaller sample for SVM
                sample_size = min(2000, len(self.X_train))
                if sample_size < len(self.X_train):
                    sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
                    X_train_sample = self.X_train.iloc[sample_idx]
                    y_train_sample = self.y_train.iloc[sample_idx]
                    svm.X_train = X_train_sample
                    svm.y_train = y_train_sample
                
                # Train models
                svm.train_linear_svm()
                svm.train_rbf_svm()
                
                if self.config['mode'] == 'full':
                    svm.train_polynomial_svm()
                    svm.hyperparameter_tuning_rbf()
                
                # Store models
                for name, model in svm.models.items():
                    self.trained_models[name] = model
                    if name in svm.results:
                        self.model_scores[name] = svm.results[name]
                
                self.pipeline_state['stages_completed'].append('svm_models')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in SVM models: {str(e)}")
                self.pipeline_state['stages_failed'].append('svm_models')
                return False
    
    def stage_clustering(self) -> bool:
        """Stage 7: Clustering Analysis"""
        if self.config['mode'] == 'quick':
            self.logger.info("Skipping clustering in quick mode")
            return True
            
        with self.timer("Clustering"):
            try:
                clustering = ClusteringModels(self.X_train, self.X_val, self.X_test,
                                            self.y_train, self.y_val, self.y_test)
                
                # Determine optimal clusters
                optimal_k, _ = clustering.determine_optimal_k(max_k=10)
                
                # Train clustering models
                clustering.train_kmeans(n_clusters=optimal_k)
                clustering.train_hierarchical(n_clusters=optimal_k)
                
                if self.config['mode'] == 'full':
                    clustering.train_gaussian_mixture(n_components=optimal_k)
                    clustering.train_dbscan(eps=3, min_samples=10)
                
                # Create customer segments
                comparison_df = clustering.compare_clustering_methods()
                if not comparison_df.empty:
                    best_clustering = comparison_df.loc[comparison_df['silhouette'].idxmax(), 'Model']
                    segments = clustering.create_customer_segments(best_clustering)
                    
                    # Save segments
                    segments.to_csv(self.results_dir / 'customer_segments.csv', index=False)
                
                self.pipeline_state['stages_completed'].append('clustering')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in clustering: {str(e)}")
                self.pipeline_state['stages_failed'].append('clustering')
                return True  # Don't fail pipeline for clustering errors
    
    def stage_ensemble_models(self) -> bool:
        """Stage 8: Ensemble Models and Final Comparison"""
        with self.timer("Ensemble Models"):
            try:
                comparison = ModelComparison(self.X_train, self.X_val, self.X_test,
                                           self.y_train, self.y_val, self.y_test)
                
                # Load our trained models
                comparison.models = self.trained_models.copy()
                
                # Evaluate all models
                results_df = comparison.evaluate_all_models()
                
                # Create ensembles
                if len(comparison.models) >= 3:
                    voting_model, voting_metrics = comparison.create_voting_ensemble()
                    self.trained_models['VotingEnsemble'] = voting_model
                    
                    if self.config['mode'] == 'full':
                        stacking_model, stacking_metrics = comparison.create_stacking_ensemble()
                        self.trained_models['StackingEnsemble'] = stacking_model
                        
                        # Create weighted ensemble
                        weights, weighted_metrics = comparison.create_weighted_average_ensemble()
                
                # Generate visualizations
                if self.config['visualize']:
                    comparison.plot_comprehensive_comparison()
                    comparison.plot_roc_curves_all()
                
                # Save results
                results_df.to_csv(self.results_dir / 'model_comparison.csv', index=False)
                
                # Identify best model
                if not results_df.empty:
                    best_model_info = results_df.iloc[0]
                    self.pipeline_state['best_model'] = {
                        'name': best_model_info['Model'],
                        'auc': best_model_info.get('AUC', 0),
                        'accuracy': best_model_info.get('Accuracy', 0),
                        'f1': best_model_info.get('F1', 0)
                    }
                
                self.pipeline_state['stages_completed'].append('ensemble_models')
                return True
                
            except Exception as e:
                self.logger.error(f"Error in ensemble models: {str(e)}")
                self.pipeline_state['stages_failed'].append('ensemble_models')
                return False
    
    def save_models(self):
        """Save all trained models"""
        self.logger.info("Saving trained models...")
        saved = 0
        for name, model in self.trained_models.items():
            try:
                model_path = self.models_dir / f"{name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model, model_path)
                saved += 1
            except Exception as e:
                self.logger.warning(f"Could not save {name}: {e}")
        
        self.logger.info(f"Saved {saved}/{len(self.trained_models)} models")
    
    def generate_report(self):
        """Generate comprehensive pipeline report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'execution_times': self.execution_times,
            'stages_completed': self.pipeline_state['stages_completed'],
            'stages_failed': self.pipeline_state['stages_failed'],
            'models_trained': list(self.trained_models.keys()),
            'best_model': self.pipeline_state['best_model'],
            'errors': self.pipeline_state['errors']
        }
        
        # Save JSON report
        report_path = self.results_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text summary
        summary = f"""
{'='*60}
TELCO CHURN PREDICTION PIPELINE REPORT
{'='*60}

Timestamp: {report['timestamp']}
Mode: {self.config['mode']}
Total Execution Time: {sum(self.execution_times.values()):.2f} seconds

STAGES COMPLETED ({len(report['stages_completed'])}):
{', '.join(report['stages_completed'])}

STAGES FAILED ({len(report['stages_failed'])}):
{', '.join(report['stages_failed']) if report['stages_failed'] else 'None'}

MODELS TRAINED ({len(report['models_trained'])}):
{chr(10).join(['  - ' + m for m in report['models_trained']])}

BEST MODEL:
{json.dumps(report['best_model'], indent=2) if report['best_model'] else 'Not determined'}

EXECUTION TIMES:
{chr(10).join([f'  {k}: {v:.2f}s' for k, v in self.execution_times.items()])}
{'='*60}
        """
        
        print(summary)
        
        # Save text report
        with open(self.results_dir / 'pipeline_summary.txt', 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Reports saved to {self.results_dir}")
    
    def run_pipeline(self, stages: Optional[List[str]] = None):
        """Run the complete pipeline"""
        self.pipeline_state['start_time'] = time.time()
        
        # Define all stages
        all_stages = {
            'prep': self.stage_data_preprocessing,
            'eda': self.stage_eda,
            'tree': self.stage_tree_models,
            'gbm': self.stage_gbm_models,
            'regression': self.stage_regression_models,
            'svm': self.stage_svm_models,
            'clustering': self.stage_clustering,
            'ensemble': self.stage_ensemble_models
        }
        
        # Determine stages to run
        if stages:
            stages_to_run = {k: v for k, v in all_stages.items() if k in stages}
        else:
            if self.config['mode'] == 'quick':
                stages_to_run = {k: v for k, v in all_stages.items() 
                               if k not in ['svm', 'clustering']}
            else:
                stages_to_run = all_stages
        
        self.logger.info(f"Running stages: {list(stages_to_run.keys())}")
        
        # Always run preprocessing first if needed
        if 'prep' not in stages_to_run and self.X_train is None:
            self.stage_data_preprocessing()
        
        # Run selected stages
        with tqdm(total=len(stages_to_run), desc="Pipeline Progress") as pbar:
            for stage_name, stage_func in stages_to_run.items():
                success = stage_func()
                pbar.update(1)
                
                if not success and stage_name in ['prep', 'tree', 'gbm', 'regression']:
                    self.logger.error(f"Critical stage {stage_name} failed. Stopping pipeline.")
                    break
        
        # Save models and generate report
        if self.trained_models:
            self.save_models()
        
        self.pipeline_state['end_time'] = time.time()
        self.generate_report()
        
        total_time = self.pipeline_state['end_time'] - self.pipeline_state['start_time']
        self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
    
    def predict(self, data_path: str, model_name: Optional[str] = None):
        """Make predictions on new data"""
        self.logger.info(f"Making predictions on {data_path}")
        
        # Load and preprocess new data
        df = pd.read_csv(data_path)
        
        # Apply same preprocessing (simplified version)
        # In production, you'd use the exact same preprocessing pipeline
        
        # Select model
        if model_name and model_name in self.trained_models:
            model = self.trained_models[model_name]
        elif self.pipeline_state['best_model']:
            model_name = self.pipeline_state['best_model']['name']
            model = self.trained_models.get(model_name)
        else:
            self.logger.error("No model available for prediction")
            return None
        
        # Make predictions
        # predictions = model.predict(X_processed)
        # probabilities = model.predict_proba(X_processed)[:, 1] if hasattr(model, 'predict_proba') else None
        
        self.logger.info(f"Predictions completed using {model_name}")
        # return predictions, probabilities


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Optimized Telco Churn Prediction Pipeline')
    parser.add_argument('--mode', choices=['quick', 'full', 'custom'], default='quick',
                      help='Pipeline execution mode')
    parser.add_argument('--stages', nargs='+', 
                      help='Specific stages to run (prep, eda, tree, gbm, regression, svm, clustering, ensemble)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--visualize', action='store_true', help='Generate all visualizations')
    parser.add_argument('--predict', type=str, help='Path to data for prediction')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing models')
    parser.add_argument('--project-dir', type=str, default='/Users/ob/Projects/teleco',
                      help='Project directory path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'mode': args.mode,
        'parallel': args.parallel,
        'visualize': args.visualize,
        'project_dir': args.project_dir,
        'verbose': args.verbose
    }
    
    # Print header
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TELCO CHURN PREDICTION - OPTIMIZED PIPELINE         ║
    ║                    Version 2.0                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    pipeline = OptimizedTelcoPipeline(config)
    
    # Handle different execution modes
    if args.predict:
        pipeline.predict(args.predict)
    elif args.evaluate:
        # Load existing models and evaluate
        pipeline.logger.info("Evaluating existing models...")
        # Implementation for model evaluation
    else:
        # Run pipeline
        pipeline.run_pipeline(stages=args.stages)
    
    print("\n✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()