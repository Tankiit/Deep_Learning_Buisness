import mlflow
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Change to your server if needed
mlflow.set_experiment("amazon_reviews_analysis")

# Load a small subset of the Beauty reviews for demonstration
def load_amazon_reviews(file_path, max_reviews=5000):
    """Load Amazon reviews from a jsonl file"""
    reviews = []
    count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if count >= max_reviews:
                break
            review = json.loads(line.strip())
            reviews.append(review)
            count += 1
    
    return pd.DataFrame(reviews)

# Define text preprocessing function
def preprocess_text(text):
    """Clean and preprocess review text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join back into string
    text = ' '.join(tokens)
    return text

# Main workflow with MLflow tracking
def analyze_amazon_reviews(file_path, category_name):
    """Run end-to-end analysis with MLflow tracking"""
    
    # Start MLflow run for data preprocessing
    with mlflow.start_run(run_name=f"{category_name}_preprocessing") as preprocessing_run:
        print("Loading and preprocessing data...")
        
        # Load data
        df = load_amazon_reviews(file_path)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("category", category_name)
        
        # Log basic dataset statistics
        mlflow.log_metric("avg_rating", df["rating"].mean())
        
        # Create binary sentiment label (positive/negative)
        df['sentiment'] = (df['rating'] >= 4).astype(int)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        mlflow.log_metric("positive_reviews", sentiment_counts.get(1, 0))
        mlflow.log_metric("negative_reviews", sentiment_counts.get(0, 0))
        
        # Track preprocessing steps
        preprocessing_steps = []
        
        # Clean review text
        preprocessing_steps.append("lowercase_conversion")
        preprocessing_steps.append("special_char_removal")
        preprocessing_steps.append("stopword_removal")
        
        # Apply preprocessing to review text
        print("Applying text preprocessing...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Log token counts
        df['token_count'] = df['processed_text'].apply(lambda x: len(x.split()))
        avg_tokens = df['token_count'].mean()
        mlflow.log_metric("avg_tokens_per_review", avg_tokens)
        
        # Log preprocessing steps
        mlflow.log_param("preprocessing_steps", preprocessing_steps)
        
        # Create visualization of token distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['token_count'], bins=50)
        plt.title('Token Count Distribution')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.savefig("token_distribution.png")
        mlflow.log_artifact("token_distribution.png")
        plt.close()
        
        # Create word clouds for positive and negative reviews
        for sentiment, label in [(1, "positive"), (0, "negative")]:
            sentiment_text = " ".join(df[df['sentiment'] == sentiment]['processed_text'])
            if sentiment_text.strip():  # Check if there's any text
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {label.capitalize()} Reviews')
                plt.savefig(f"{label}_wordcloud.png")
                mlflow.log_artifact(f"{label}_wordcloud.png")
                plt.close()
        
        # Store preprocessing run ID
        preprocessing_run_id = preprocessing_run.info.run_id
    
    # Start MLflow run for feature extraction
    with mlflow.start_run(run_name=f"{category_name}_feature_extraction") as feature_run:
        print("Extracting features...")
        
        # Link to preprocessing run
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
        
        # Set up TF-IDF vectorization
        max_features = 1000
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", max_features)
        
        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment']
        
        # Log feature information
        mlflow.log_param("vocabulary_size", len(vectorizer.vocabulary_))
        
        # Split data
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])
        
        # Store split data distribution
        train_pos = y_train.sum()
        train_neg = len(y_train) - train_pos
        test_pos = y_test.sum()
        test_neg = len(y_test) - test_pos
        
        mlflow.log_metric("train_positive", train_pos)
        mlflow.log_metric("train_negative", train_neg)
        mlflow.log_metric("test_positive", test_pos)
        mlflow.log_metric("test_negative", test_neg)
        
        # Create top features visualization
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sums = X.sum(axis=0).A1
        top_indices = tfidf_sums.argsort()[-20:][::-1]
        top_features = [(feature_names[i], tfidf_sums[i]) for i in top_indices]
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        x, y = zip(*[(f, v) for f, v in top_features])
        plt.barh(x, y)
        plt.title('Top 20 Features by TF-IDF Score')
        plt.xlabel('TF-IDF Sum')
        plt.tight_layout()
        plt.savefig("top_features.png")
        mlflow.log_artifact("top_features.png")
        plt.close()
        
        # Store feature extraction run ID
        feature_run_id = feature_run.info.run_id
    
    # Start MLflow run for model training
    with mlflow.start_run(run_name=f"{category_name}_model_training") as model_run:
        print("Training model...")
        
        # Link to feature extraction run
        mlflow.log_param("feature_extraction_run_id", feature_run_id)
        
        # Define model parameters
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
        
        # Log all parameters
        mlflow.log_params(model_params)
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        # Create feature importance visualization
        feature_importances = model.feature_importances_
        top_indices = feature_importances.argsort()[-15:][::-1]
        top_features = [(feature_names[i], feature_importances[i]) for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        x, y = zip(*[(f, v) for f, v in top_features])
        plt.barh(x, y)
        plt.title('Top 15 Features by Importance')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        # Log the model
        mlflow.sklearn.log_model(model, "sentiment_model")
        
        # Store model run ID
        model_run_id = model_run.info.run_id
        
        print(f"Model training complete with accuracy: {accuracy:.4f}")
        print(f"Run IDs:")
        print(f"  Preprocessing: {preprocessing_run_id}")
        print(f"  Feature Extraction: {feature_run_id}")
        print(f"  Model Training: {model_run_id}")
        
        return {
            "preprocessing_run_id": preprocessing_run_id,
            "feature_run_id": feature_run_id,
            "model_run_id": model_run_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

# Run the analysis for Beauty reviews
if __name__ == "__main__":
    # Change this path to your data file location
    file_path = "All_Beauty.jsonl"  # Adjust to your actual path
    category = "Beauty"
    
    results = analyze_amazon_reviews(file_path, category)
    
    print("\nMLflow tracking complete! Open the MLflow UI to view the results.")
    print("Run: 'mlflow ui' in your terminal if not already running")
    
    # Optional: Compare with another model
    with mlflow.start_run(run_name=f"{category}_optimized_model") as run:
        # Link to the original preprocessing and feature runs
        mlflow.log_param("preprocessing_run_id", results["preprocessing_run_id"])
        mlflow.log_param("feature_run_id", results["feature_run_id"])
        
        # Log improved model parameters
        improved_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "random_state": 42
        }
        mlflow.log_params(improved_params)
        
        # Log simulated better metrics (in a real scenario, you would train a new model)
        # This is just for demonstration
        mlflow.log_metric("accuracy", results["accuracy"] + 0.03)
        mlflow.log_metric("precision", results["precision"] + 0.05)
        mlflow.log_metric("recall", results["recall"] + 0.02)
        mlflow.log_metric("f1_score", results["f1"] + 0.03)
        
        print("Added comparison model for demonstration purposes")