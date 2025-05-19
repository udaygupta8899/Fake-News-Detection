# End-to-End MLOps Project Pipeline Documentation

This guide outlines an end-to-end MLOps pipeline setup using the following tools and platforms:

* **DVC** for data and model versioning
* **MLflow** for experiment tracking and model registry
* **DagsHub** for Git and DVC remote repository
* **GitHub Actions** for CI/CD automation
* **Docker** and **DockerHub** for containerization
* **AWS S3**, **ECR**, and **EKS** for production deployment
* **Prometheus** and **Grafana** for monitoring

> **Note:** Replace the project name, bucket names, and AWS credentials according to your own.

---

## 🛠️ 1. Initial Setup

### Create Project Structure

```bash
mkdir my-mlops-project && cd my-mlops-project
mkdir data notebooks src models
```

### Initialize Git and Python Environment

```bash
git init
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -U pip setuptools
pip install dvc mlflow pandas scikit-learn
```

Create a `requirements.txt`:

```txt
dvc
mlflow
pandas
scikit-learn
```

---

## 📦 2. DVC Setup

### Initialize DVC

```bash
dvc init
dvc remote add -d myremote https://dagshub.com/<username>/<repo>.dvc
```

### Track Dataset

```bash
mkdir data/raw
dvc add data/raw
```

Add and push:

```bash
git add data/raw.dvc .gitignore dvc.yaml dvc.lock

git commit -m "Add raw data with DVC"
dvc push
```

---

## 🧪 3. MLflow for Experiment Tracking

### Set MLflow Tracking URI (e.g., local or remote)

```python
import mlflow
mlflow.set_tracking_uri("file:///absolute/path/to/mlruns")
mlflow.set_experiment("fake-news-detection")
```

### Example Training Script

```python
# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    mlflow.sklearn.log_model(clf, "model")
```

Run and Track:

```bash
python src/train.py
```

---

## 🔁 4. DVC Pipelines

Create pipeline stages using DVC:

```bash
dvc stage add -n preprocess -d src/preprocess.py -o data/processed python src/preprocess.py

dvc stage add -n train -d src/train.py -d data/processed -o models/model.pkl python src/train.py

dvc repro
```

Push changes:

```bash
dvc push
git add dvc.yaml dvc.lock
git commit -m "Add DVC pipeline stages"
```

---

## 🚀 5. CI/CD with GitHub Actions

### `.github/workflows/train.yml`

```yaml
name: Model Training Pipeline

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Training
        run: |
          dvc pull
          dvc repro
          dvc push
```

---

## 🐳 6. Dockerization

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
```

Build and push image:

```bash
docker build -t myuser/fake-news-detector .
docker push myuser/fake-news-detector
```

---

## ☁️ 7. AWS Deployment

### Setup

```bash
aws configure  # enter keys, region

# Create S3 bucket for models
aws s3 mb s3://my-mlops-bucket

# Push model
aws s3 cp models/model.pkl s3://my-mlops-bucket/
```

### ECR (Elastic Container Registry)

```bash
ecr_uri=123456789012.dkr.ecr.us-east-1.amazonaws.com/fake-news-app
aws ecr create-repository --repository-name fake-news-app

# Authenticate Docker to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ecr_uri

docker tag myuser/fake-news-detector $ecr_uri

docker push $ecr_uri
```

### EKS Setup

* Use eksctl or Terraform to set up EKS cluster.

```bash
eksctl create cluster --name mlops-cluster --region us-east-1
```

Deploy with Kubernetes manifests (e.g. `deployment.yaml`, `service.yaml`)

Apply to cluster:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 📊 8. Monitoring with Prometheus + Grafana

### Add Prometheus

```yaml
# prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
...
```

Expose Prometheus as a service.

### Add Grafana

```yaml
# grafana-deployment.yaml
apiVersion: apps/v1
kind: Deployment
...
```

Connect Prometheus as a data source in Grafana dashboard.

---

## 🔄 9. Automation (Optional)

Create scheduled retraining using GitHub Actions with a cron trigger:

```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # every Sunday at midnight
```

---

## ✅ Final Notes

* Use `.env` files for secrets or GitHub secrets
* Backup DVC, models, and metrics
* Monitor performance drift using Prometheus metrics
