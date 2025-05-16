import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from optuna.integration.mlflow import MLflowCallback

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# --------------------------
# Configuration / DagsHub
# --------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/udaygupta8899/Fake-News-Detection.mlflow"
dagshub.init(
    repo_owner="udaygupta8899",
    repo_name="Fake-News-Detection",
    mlflow=True
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RFC Hyperparameter Tuning")

# --------------------------
# Text Preprocessing
# --------------------------
def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(tokens)

# --------------------------
# Data Loading & Vectorizing
# --------------------------
def load_and_prepare_data(filepath: str):
    df = pd.read_csv(filepath)
    df["title"] = df["title"].astype(str).apply(preprocess_text)
    df["text"]  = df["text"].astype(str).apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[["title", "text"]].agg(" ".join, axis=1))
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

# --------------------------
# Optuna Objective
# --------------------------
def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(
        rf, X_train, y_train, cv=5,
        scoring="f1", n_jobs=-1, error_score="raise"
    )
    return scores.mean()

# --------------------------
# Training & MLflow Logging
# --------------------------
def train_and_log_model_rf(X_train, X_test, y_train, y_test):
    """
    Runs an Optuna study (with MLflowCallback) to tune RF—each trial
    appears as its own MLflow run—and then logs the final best model
    and its test metrics in a separate MLflow run.
    """
    # This callback will start and end runs for each trial automatically.
    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="f1_cv_mean"
    )

    # 1) Tune hyperparameters—each Optuna trial => MLflow run
    study = optuna.create_study(direction="maximize", study_name="rf_optuna_study")
    study.optimize(
        lambda t: objective(t, X_train, y_train),
        n_trials=50,
        callbacks=[mlflow_cb]
    )

    # 2) After tuning, log best params & final model under a new run
    best_params = study.best_params
    best_score  = study.best_value

    with mlflow.start_run(run_name="RF Final Model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1", best_score)

        # Retrain on full training set
        best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        best_rf.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = best_rf.predict(X_test)
        test_metrics = {
            "accuracy":  accuracy_score(y_test,   y_pred),
            "precision": precision_score(y_test,  y_pred),
            "recall":    recall_score(y_test,     y_pred),
            "f1_score":  f1_score(y_test,         y_pred),
        }
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest_model")

        print(f"Best CV F1: {best_score:.4f}")
        print("Test performance:")
        print(
            f"  Acc: {test_metrics['accuracy']:.4f} | "
            f"Prec: {test_metrics['precision']:.4f} | "
            f"Rec: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1_score']:.4f}"
        )


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(
        "notebooks/data_cleaned.csv"
    )
    train_and_log_model_rf(X_train, X_test, y_train, y_test)
