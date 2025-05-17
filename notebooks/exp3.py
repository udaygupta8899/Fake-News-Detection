import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/udaygupta8899/Fake-News-Detection.mlflow"
dagshub.init(repo_owner="udaygupta8899", repo_name="Fake-News-Detection", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RandomForest Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df["title"] = df["title"].astype(str).apply(preprocess_text)
    df["text"]  = df["text"].astype(str).apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[["title", "text"]].agg(" ".join, axis=1))
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        for params, mean_score, std_score in zip(grid_search.cv_results_["params"],
                                                 grid_search.cv_results_["mean_test_score"],
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"RF with params: {params}", nested=True):
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data_cleaned.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)