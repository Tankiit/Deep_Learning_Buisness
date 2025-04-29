import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Set the MLflow tracking server
# For a local experiment, you can skip this line to use the default local directory
# mlflow.set_tracking_uri("http://localhost:5000")

# Create a new experiment (or use an existing one)
experiment_name = "kmeans_demo"
mlflow.set_experiment(experiment_name)

# Generate synthetic data with clear clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Utility function to plot the clusters
def plot_clusters(X, labels, centers=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.8)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-means Clustering Results')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

# Try different values of k and track the results with MLflow
for k in range(2, 7):
    # Start a new MLflow run
    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        print(f"Running K-means with k={k}")
        
        # Log the parameter
        mlflow.log_param("k", k)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_init", 10)
        
        # Train K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Get results
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, labels) if k > 1 else 0
        
        # Log metrics
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("silhouette_score", silhouette)
        
        # Log cluster distribution
        for i in range(k):
            cluster_size = np.sum(labels == i)
            mlflow.log_metric(f"cluster_{i}_size", cluster_size)
        
        # Create and log visualization
        fig = plot_clusters(X, labels, centers)
        mlflow.log_figure(fig, f"kmeans_k{k}_clusters.png")
        plt.close(fig)
        
        # Log the model
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette Score: {silhouette:.2f}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        print("---")

# After all runs are complete, find the best model based on silhouette score
client = mlflow.tracking.MlflowClient()
runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    order_by=["metrics.silhouette_score DESC"]
)

# Display the best run
if not runs.empty:
    best_run = runs.iloc[0]
    best_k = int(best_run["params.k"])
    best_silhouette = best_run["metrics.silhouette_score"]
    best_run_id = best_run["run_id"]
    
    print(f"Best model: k={best_k} with silhouette score {best_silhouette:.2f}")
    print(f"Run ID: {best_run_id}")
    
    # Optional: Register the best model in the model registry
    model_uri = f"runs:/{best_run_id}/kmeans_model"
    registered_model = mlflow.register_model(model_uri, "best_kmeans_model")
    print(f"Registered model version: {registered_model.version}")
else:
    print("No runs found.")

print("\nMLflow tracking complete! Open the MLflow UI to view the results.")
print("Run: 'mlflow ui' in your terminal if not already running")