import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris(as_frame=True)
X = iris.data

# Start MLflow run
with mlflow.start_run(run_name="better_feature_tracking"):

    # Log small things as params
    mlflow.log_param("raw_data_shape", X.shape)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_feature_names = [f"scaled_{name}" for name in X.columns]

    # Create DataFrame for scaled features
    X_scaled_df = pd.DataFrame(X_scaled, columns=scaled_feature_names)

    # Save the feature names
    with open("scaled_feature_names.txt", "w") as f:
        for name in scaled_feature_names:
            f.write(f"{name}\n")
    mlflow.log_artifact("scaled_feature_names.txt")

    # Save a sample of the scaled features
    X_scaled_df.head(5).to_csv("scaled_features_sample.csv", index=False)
    mlflow.log_artifact("scaled_features_sample.csv")