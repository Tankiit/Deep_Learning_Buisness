import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Set up GridSearchCV with cross-validation
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)

# Start MLflow run
with mlflow.start_run(run_name="rf_gridsearch_cv"):
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best parameters
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_cv_accuracy", best_score)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)

    # Cross-validated metrics (on train set)
    cv_acc = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_prec = cross_val_score(best_model, X_train, y_train, cv=5, scoring=make_scorer(precision_score, average='macro'))
    cv_rec = cross_val_score(best_model, X_train, y_train, cv=5, scoring=make_scorer(recall_score, average='macro'))
    mlflow.log_metric("cv_accuracy_mean", np.mean(cv_acc))
    mlflow.log_metric("cv_precision_mean", np.mean(cv_prec))
    mlflow.log_metric("cv_recall_mean", np.mean(cv_rec))

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # Feature Importance plot
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x="importance", y="feature", data=feature_importance, ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

    # Save best model
    mlflow.sklearn.log_model(best_model, "random_forest_model", input_example=X_test.iloc[:1])

n_estimators_list = [10, 20, 50, 100, 200]
for n in n_estimators_list:
    with mlflow.start_run(run_name=f"rf_n_estimators_{n}"):
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)