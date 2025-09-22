# clustering_models.py - Clustering Analysis for Telco Customer Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                             MeanShift, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                           calinski_harabasz_score, adjusted_rand_score,
                           normalized_mutual_info_score)
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
import os
warnings.filterwarnings('ignore')

# Import preprocessing functions
from prep import get_preprocessed_data

class ClusteringModels:
    """Class for customer segmentation using clustering algorithms"""
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Combine train and val for clustering
        self.X_full = pd.concat([X_train, X_val])
        self.y_full = pd.concat([y_train, y_val])
        
        self.models = {}
        self.labels = {}
        self.metrics = {}
        self.cluster_profiles = {}
        
    def determine_optimal_k(self, max_k=15):
        """Determine optimal number of clusters using elbow method and silhouette score"""
        print("\nDetermining optimal number of clusters...")
        
        inertias = []
        silhouettes = []
        db_scores = []
        ch_scores = []
        
        K = range(2, max_k)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_full)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.X_full, labels))
            db_scores.append(davies_bouldin_score(self.X_full, labels))
            ch_scores.append(calinski_harabasz_score(self.X_full, labels))
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Elbow curve
        axes[0, 0].plot(K, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[0, 1].plot(K, silhouettes, 'go-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (Higher is Better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Davies-Bouldin Index
        axes[1, 0].plot(K, db_scores, 'ro-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Davies-Bouldin Index')
        axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Index
        axes[1, 1].plot(K, ch_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Calinski-Harabasz Index')
        axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Cluster Evaluation Metrics', fontsize=14)
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/optimal_clusters_analysis.png")
        plt.show()
        
        # Determine optimal k
        optimal_k = K[silhouettes.index(max(silhouettes))]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, {'inertias': inertias, 'silhouettes': silhouettes, 
                          'db_scores': db_scores, 'ch_scores': ch_scores}
    
    def train_kmeans(self, n_clusters=None):
        """Train K-Means clustering"""
        if n_clusters is None:
            n_clusters, _ = self.determine_optimal_k()
        
        print(f"\nTraining K-Means with {n_clusters} clusters...")
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(self.X_full)
        
        self.models['KMeans'] = model
        self.labels['KMeans'] = labels
        
        # Evaluate
        metrics = self.evaluate_clustering(labels, 'KMeans')
        
        # Analyze clusters
        self.analyze_clusters(labels, 'KMeans')
        
        return model, labels, metrics
    
    def train_dbscan(self, eps=0.5, min_samples=5):
        """Train DBSCAN clustering"""
        print(f"\nTraining DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X_full)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        self.models['DBSCAN'] = model
        self.labels['DBSCAN'] = labels
        
        # Evaluate (if we have valid clusters)
        if n_clusters > 1:
            metrics = self.evaluate_clustering(labels, 'DBSCAN')
        else:
            metrics = {'n_clusters': n_clusters, 'n_noise': n_noise}
        
        return model, labels, metrics
    
    def train_hierarchical(self, n_clusters=None):
        """Train Hierarchical/Agglomerative clustering"""
        if n_clusters is None:
            n_clusters, _ = self.determine_optimal_k()
        
        print(f"\nTraining Hierarchical Clustering with {n_clusters} clusters...")
        
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(self.X_full)
        
        self.models['Hierarchical'] = model
        self.labels['Hierarchical'] = labels
        
        # Evaluate
        metrics = self.evaluate_clustering(labels, 'Hierarchical')
        
        # Analyze clusters
        self.analyze_clusters(labels, 'Hierarchical')
        
        return model, labels, metrics
    
    def train_gaussian_mixture(self, n_components=None):
        """Train Gaussian Mixture Model"""
        if n_components is None:
            n_components, _ = self.determine_optimal_k()
        
        print(f"\nTraining Gaussian Mixture Model with {n_components} components...")
        
        model = GaussianMixture(n_components=n_components, random_state=42)
        model.fit(self.X_full)
        labels = model.predict(self.X_full)
        
        self.models['GaussianMixture'] = model
        self.labels['GaussianMixture'] = labels
        
        # Evaluate
        metrics = self.evaluate_clustering(labels, 'GaussianMixture')
        
        # Add BIC and AIC
        metrics['BIC'] = model.bic(self.X_full)
        metrics['AIC'] = model.aic(self.X_full)
        
        print(f"BIC: {metrics['BIC']:.2f}")
        print(f"AIC: {metrics['AIC']:.2f}")
        
        # Analyze clusters
        self.analyze_clusters(labels, 'GaussianMixture')
        
        return model, labels, metrics
    
    def train_spectral(self, n_clusters=None):
        """Train Spectral Clustering"""
        if n_clusters is None:
            n_clusters, _ = self.determine_optimal_k()
        
        print(f"\nTraining Spectral Clustering with {n_clusters} clusters...")
        
        # Use subsample for computational efficiency
        sample_size = min(2000, len(self.X_full))
        X_sample = self.X_full.sample(n=sample_size, random_state=42)
        
        model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', 
                                  random_state=42)
        labels_sample = model.fit_predict(X_sample)
        
        # For full dataset, use KMeans on the sample centroids
        kmeans_full = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans_full.fit_predict(self.X_full)
        
        self.models['Spectral'] = model
        self.labels['Spectral'] = labels
        
        # Evaluate
        metrics = self.evaluate_clustering(labels, 'Spectral')
        
        return model, labels, metrics
    
    def evaluate_clustering(self, labels, model_name):
        """Evaluate clustering performance"""
        # Filter out noise points for metrics that don't handle them
        valid_mask = labels != -1
        labels_valid = labels[valid_mask]
        X_valid = self.X_full[valid_mask]
        
        metrics = {}
        
        if len(set(labels_valid)) > 1:
            # Internal metrics
            metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
            metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
            
            # External metrics (comparing with actual churn labels)
            y_valid = self.y_full[valid_mask]
            metrics['ari'] = adjusted_rand_score(y_valid, labels_valid)
            metrics['nmi'] = normalized_mutual_info_score(y_valid, labels_valid)
        
        metrics['n_clusters'] = len(set(labels_valid))
        
        print(f"\n{model_name} Metrics:")
        for metric, value in metrics.items():
            if metric != 'n_clusters':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        self.metrics[model_name] = metrics
        
        return metrics
    
    def analyze_clusters(self, labels, model_name):
        """Analyze cluster characteristics"""
        print(f"\nAnalyzing {model_name} clusters...")
        
        # Create cluster DataFrame
        cluster_df = self.X_full.copy()
        cluster_df['Cluster'] = labels
        cluster_df['Churn'] = self.y_full.values
        
        # Calculate cluster profiles
        profiles = {}
        
        for cluster in sorted(set(labels)):
            if cluster != -1:  # Skip noise points
                cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
                
                profile = {
                    'Size': len(cluster_data),
                    'Size_Percentage': len(cluster_data) / len(cluster_df) * 100,
                    'Churn_Rate': cluster_data['Churn'].mean() * 100
                }
                
                # Add mean values for key features
                key_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
                for feature in key_features:
                    if feature in cluster_data.columns:
                        profile[f'Mean_{feature}'] = cluster_data[feature].mean()
                
                profiles[f'Cluster_{cluster}'] = profile
        
        self.cluster_profiles[model_name] = profiles
        
        # Print summary
        print(f"\n{model_name} Cluster Summary:")
        for cluster, profile in profiles.items():
            print(f"\n{cluster}:")
            print(f"  Size: {profile['Size']} ({profile['Size_Percentage']:.1f}%)")
            print(f"  Churn Rate: {profile['Churn_Rate']:.1f}%")
            if 'Mean_tenure' in profile:
                print(f"  Mean Tenure: {profile['Mean_tenure']:.1f}")
            if 'Mean_MonthlyCharges' in profile:
                print(f"  Mean Monthly Charges: ${profile['Mean_MonthlyCharges']:.2f}")
        
        return profiles
    
    def visualize_clusters_2d(self, method='PCA'):
        """Visualize clusters in 2D space"""
        print(f"\nVisualizing clusters using {method}...")
        
        # Dimensionality reduction
        if method == 'PCA':
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(self.X_full)
            xlabel, ylabel = 'First Principal Component', 'Second Principal Component'
        elif method == 'TSNE':
            # Use smaller sample for t-SNE
            sample_size = min(2000, len(self.X_full))
            sample_indices = np.random.choice(len(self.X_full), sample_size, replace=False)
            X_sample = self.X_full.iloc[sample_indices]
            
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X_sample)
            xlabel, ylabel = 't-SNE Component 1', 't-SNE Component 2'
        
        # Plot clusters for each model
        n_models = len(self.labels)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, labels) in enumerate(self.labels.items()):
            if idx < 6:
                if method == 'TSNE':
                    labels_plot = labels[sample_indices]
                else:
                    labels_plot = labels
                
                scatter = axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                          c=labels_plot, cmap='viridis', 
                                          s=10, alpha=0.6)
                axes[idx].set_xlabel(xlabel)
                axes[idx].set_ylabel(ylabel)
                axes[idx].set_title(f'{model_name}')
                plt.colorbar(scatter, ax=axes[idx])
        
        plt.suptitle(f'Cluster Visualizations ({method})', fontsize=14)
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cluster_visualization_{method.lower()}.png")
        plt.show()
    
    def plot_dendrogram(self, max_clusters=30):
        """Plot hierarchical clustering dendrogram"""
        print("\nCreating dendrogram...")
        
        # Use subsample for dendrogram
        sample_size = min(500, len(self.X_full))
        X_sample = self.X_full.sample(n=sample_size, random_state=42)
        
        # Create linkage matrix
        linkage_matrix = linkage(X_sample, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, 
                  truncate_mode='lastp',
                  p=max_clusters,
                  leaf_rotation=90,
                  leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/dendrogram.png")
        plt.show()
    
    def plot_cluster_churn_analysis(self):
        """Analyze churn patterns across clusters"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, labels) in enumerate(self.labels.items()):
            if idx < 6:
                # Calculate churn rate per cluster
                cluster_churn = pd.DataFrame({
                    'Cluster': labels,
                    'Churn': self.y_full.values
                })
                
                churn_rates = cluster_churn.groupby('Cluster')['Churn'].agg(['mean', 'count'])
                churn_rates = churn_rates[churn_rates['count'] > 10]  # Filter small clusters
                
                # Plot
                bars = axes[idx].bar(range(len(churn_rates)), 
                                   churn_rates['mean'] * 100)
                axes[idx].set_xlabel('Cluster')
                axes[idx].set_ylabel('Churn Rate (%)')
                axes[idx].set_title(f'{model_name} - Churn by Cluster')
                axes[idx].set_xticks(range(len(churn_rates)))
                axes[idx].set_xticklabels(churn_rates.index)
                
                # Color bars based on churn rate
                colors = plt.cm.RdYlGn_r(churn_rates['mean'])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                # Add cluster sizes as text
                for i, (idx_val, row) in enumerate(churn_rates.iterrows()):
                    axes[idx].text(i, row['mean'] * 100 + 1, 
                                 f"n={int(row['count'])}", 
                                 ha='center', fontsize=8)
        
        plt.suptitle('Churn Rate Analysis by Cluster', fontsize=14)
        plt.tight_layout()
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cluster_churn_analysis.png")
        plt.show()
    
    def plot_cluster_feature_profiles(self, model_name='KMeans'):
        """Plot feature profiles for each cluster"""
        if model_name not in self.labels:
            print(f"Model {model_name} not found.")
            return
        
        labels = self.labels[model_name]
        
        # Select key features to profile
        features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        available_features = [f for f in features if f in self.X_full.columns]
        
        if not available_features:
            print("No suitable features for profiling.")
            return
        
        # Create cluster DataFrame
        cluster_df = self.X_full[available_features].copy()
        cluster_df['Cluster'] = labels
        
        # Calculate mean values per cluster
        cluster_means = cluster_df.groupby('Cluster').mean()
        
        # Normalize for visualization
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means_norm.T, annot=True, fmt='.2f', 
                   cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'})
        plt.title(f'{model_name} - Cluster Feature Profiles')
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        
        output_dir = "data/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cluster_profiles_{model_name}.png")
        plt.show()
        
        # Print actual values
        print(f"\n{model_name} Cluster Feature Means:")
        print(cluster_means)
    
    def compare_clustering_methods(self):
        """Compare all clustering methods"""
        comparison_data = []
        
        for model_name, metrics in self.metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        metrics_to_plot = ['silhouette', 'davies_bouldin', 'ari', 'nmi']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(available_metrics):
                data = comparison_df.dropna(subset=[metric])
                bars = axes[idx].bar(range(len(data)), data[metric])
                axes[idx].set_xticks(range(len(data)))
                axes[idx].set_xticklabels(data['Model'], rotation=45, ha='right')
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].set_title(f'{metric.replace("_", " ").title()}')
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                # Color best performer
                if metric in ['silhouette', 'ari', 'nmi', 'calinski_harabasz']:
                    best_idx = data[metric].idxmax()
                else:  # davies_bouldin - lower is better
                    best_idx = data[metric].idxmin()
                
                best_pos = data.index.get_loc(best_idx)
                bars[best_pos].set_color('green')
            
            plt.suptitle('Clustering Methods Comparison', fontsize=14)
            plt.tight_layout()
            
            output_dir = "data/plots"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/clustering_comparison.png")
            plt.show()
        
        return comparison_df
    
    def create_customer_segments(self, model_name='KMeans'):
        """Create actionable customer segments based on clustering"""
        if model_name not in self.labels:
            print(f"Model {model_name} not found.")
            return None
        
        labels = self.labels[model_name]
        
        # Create segmentation DataFrame
        segments_df = self.X_full.copy()
        segments_df['Cluster'] = labels
        segments_df['Churn'] = self.y_full.values
        
        # Analyze each segment
        segment_analysis = []
        
        for cluster in sorted(set(labels)):
            if cluster != -1:
                segment_data = segments_df[segments_df['Cluster'] == cluster]
                
                analysis = {
                    'Segment': f'Segment_{cluster}',
                    'Size': len(segment_data),
                    'Percentage': len(segment_data) / len(segments_df) * 100,
                    'Churn_Rate': segment_data['Churn'].mean() * 100,
                    'Avg_Tenure': segment_data['tenure'].mean() if 'tenure' in segment_data.columns else 0,
                    'Avg_MonthlyCharges': segment_data['MonthlyCharges'].mean() if 'MonthlyCharges' in segment_data.columns else 0,
                    'Avg_TotalCharges': segment_data['TotalCharges'].mean() if 'TotalCharges' in segment_data.columns else 0,
                }
                
                # Determine segment characteristics
                if analysis['Churn_Rate'] > 40:
                    analysis['Risk_Level'] = 'High'
                    analysis['Recommendation'] = 'Immediate retention campaign'
                elif analysis['Churn_Rate'] > 25:
                    analysis['Risk_Level'] = 'Medium'
                    analysis['Recommendation'] = 'Proactive engagement'
                else:
                    analysis['Risk_Level'] = 'Low'
                    analysis['Recommendation'] = 'Maintain service quality'
                
                # Customer value assessment
                if analysis['Avg_TotalCharges'] > segments_df['TotalCharges'].median():
                    analysis['Value'] = 'High Value'
                else:
                    analysis['Value'] = 'Standard Value'
                
                segment_analysis.append(analysis)
        
        segments_results = pd.DataFrame(segment_analysis)
        
        print(f"\n{model_name} Customer Segmentation Results:")
        print(segments_results.to_string(index=False))
        
        # Save results
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        segments_results.to_csv(f"{output_dir}/customer_segments_{model_name}.csv", index=False)
        
        return segments_results

def main():
    """Main execution function"""
    print("="*60)
    print("CLUSTERING MODELS FOR TELCO CUSTOMER SEGMENTATION")
    print("="*60)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    # Initialize clustering models class
    clustering = ClusteringModels(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Determine optimal number of clusters
    print("\n" + "-"*40)
    print("DETERMINING OPTIMAL CLUSTERS")
    print("-"*40)
    optimal_k, k_metrics = clustering.determine_optimal_k()
    
    # Train different clustering algorithms
    print("\n" + "-"*40)
    print("TRAINING CLUSTERING MODELS")
    print("-"*40)
    
    # 1. K-Means
    kmeans_model, kmeans_labels, kmeans_metrics = clustering.train_kmeans(n_clusters=optimal_k)
    
    # 2. DBSCAN
    dbscan_model, dbscan_labels, dbscan_metrics = clustering.train_dbscan(eps=3, min_samples=10)
    
    # 3. Hierarchical Clustering
    hier_model, hier_labels, hier_metrics = clustering.train_hierarchical(n_clusters=optimal_k)
    
    # 4. Gaussian Mixture Model
    gmm_model, gmm_labels, gmm_metrics = clustering.train_gaussian_mixture(n_components=optimal_k)
    
    # 5. Spectral Clustering
    spectral_model, spectral_labels, spectral_metrics = clustering.train_spectral(n_clusters=optimal_k)
    
    # Visualizations
    print("\n" + "-"*40)
    print("CREATING VISUALIZATIONS")
    print("-"*40)
    
    # Dendrogram for hierarchical clustering
    clustering.plot_dendrogram()
    
    # 2D visualizations
    clustering.visualize_clusters_2d(method='PCA')
    
    # Churn analysis by cluster
    clustering.plot_cluster_churn_analysis()
    
    # Feature profiles for best model
    clustering.plot_cluster_feature_profiles('KMeans')
    
    # Compare methods
    print("\n" + "-"*40)
    print("COMPARING CLUSTERING METHODS")
    print("-"*40)
    comparison_results = clustering.compare_clustering_methods()
    
    # Create customer segments
    print("\n" + "-"*40)
    print("CREATING CUSTOMER SEGMENTS")
    print("-"*40)
    
    # Create segments using best performing model
    best_model = comparison_results.loc[comparison_results['silhouette'].idxmax(), 'Model']
    print(f"\nUsing {best_model} for final segmentation...")
    segments = clustering.create_customer_segments(best_model)
    
    # Save results
    print("\n" + "-"*40)
    print("SAVING RESULTS")
    print("-"*40)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    comparison_results.to_csv(f"{output_dir}/clustering_comparison.csv", index=False)
    print(f"Comparison results saved to {output_dir}/clustering_comparison.csv")
    
    # Save cluster assignments
    for model_name, labels in clustering.labels.items():
        assignments_df = pd.DataFrame({
            'CustomerIndex': range(len(labels)),
            'Cluster': labels
        })
        assignments_df.to_csv(f"{output_dir}/cluster_assignments_{model_name}.csv", index=False)
    
    # Summary
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"Best performing model: {best_model}")
    print("\nTop Clustering Methods by Silhouette Score:")
    print(comparison_results[['Model', 'silhouette', 'n_clusters']].sort_values(
        'silhouette', ascending=False).head(3).to_string(index=False))
    
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()