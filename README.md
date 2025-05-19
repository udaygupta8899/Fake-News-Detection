---

# 📰 Fake News Detection

## 🔍 Overview

In an era where misinformation spreads rapidly, the ability to detect fake news is crucial. This project presents a robust pipeline for detecting fake news articles using machine learning techniques. It encompasses data preprocessing, model training, evaluation, and deployment, ensuring a comprehensive approach to the problem.([GitHub][1])

## 📁 Project Structure

The repository follows a modular and organized structure inspired by best practices in machine learning project development:

```
Fake-News-Detection/
├── .dvc/                   # DVC configuration files
├── .github/workflows/      # GitHub Actions workflows for CI/CD
├── docs/                   # Documentation files
├── flask_app/              # Flask application for deployment
├── models/                 # Serialized models and model checkpoints
├── notebooks/              # Jupyter notebooks for exploration and experimentation
├── references/             # Data dictionaries, manuals, and all other explanatory materials
├── reports/                # Generated analysis as HTML, PDF, LaTeX, etc.
├── scripts/                # Utility scripts for various tasks
├── src/                    # Source code for use in this project
│   ├── data/               # Scripts to download or generate data
│   ├── features/           # Scripts to turn raw data into features for modeling
│   ├── models/             # Scripts to train models and then use trained models to make predictions
│   └── visualization/      # Scripts to create exploratory and results-oriented visualizations
├── tests/                  # Unit tests and test datasets
├── .dvcignore              # DVC ignore file
├── .gitignore              # Git ignore file
├── Dockerfile              # Dockerfile for containerization
├── LICENSE                 # License file
├── Makefile                # Makefile with commands like `make data` or `make train`
├── README.md               # Project README
├── deployment.yaml         # Kubernetes deployment configuration
├── dvc.lock                # DVC lock file
├── dvc.yaml                # DVC pipeline configuration
├── params.yaml             # Parameters for experiments
├── requirements.txt        # Python dependencies
├── setup.py                # Makes project pip installable
├── test_environment.py     # Script to test the environment setup
└── tox.ini                 # Tox configuration file
```



## 🚀 Getting Started

### Prerequisites

* Python 3.10
* [Conda](https://docs.conda.io/en/latest/)
* [Git](https://git-scm.com/)
* [Docker](https://www.docker.com/)
* [AWS CLI](https://aws.amazon.com/cli/)
* [kubectl](https://kubernetes.io/docs/tasks/tools/)
* [eksctl](https://eksctl.io/)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/udaygupta8899/Fake-News-Detection.git
   cd Fake-News-Detection
   ```



2. **Create and Activate Virtual Environment**

   ```bash
   conda create -n atlas python=3.10
   conda activate atlas
   ```



3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```



4. **Install Cookiecutter Template**

   ```bash
   pip install cookiecutter
   cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
   ```



5. **Initialize DVC**

   ```bash
   dvc init
   ```



## 🧪 Experiment Tracking with MLflow and DVC

1. **Set Up MLflow with DagsHub**

   * Create a new repository on [DagsHub](https://dagshub.com/).
   * Connect your GitHub repository to DagsHub.
   * Copy the MLflow tracking URI provided by DagsHub.

2. **Configure MLflow**

   ```bash
   pip install mlflow dagshub
   export MLFLOW_TRACKING_URI=<your_dagshub_tracking_uri>
   ```



3. **Run Experiments**

   Execute your Jupyter notebooks or scripts to log experiments to MLflow.

4. **Track Data and Models with DVC**

   ```bash
   dvc add data/raw
   dvc add models/
   git add data/raw.dvc models.dvc .gitignore
   git commit -m "Add raw data and models to DVC"
   ```



## 🛠️ Model Development Pipeline

1. **Data Ingestion**

   Scripts located in `src/data/` handle data loading and initial preprocessing.

2. **Data Preprocessing**

   Further cleaning and preparation are performed using scripts in `src/features/`.

3. **Feature Engineering**

   Feature extraction and selection methods are implemented in `src/features/`.

4. **Model Training**

   Training scripts utilizing various algorithms are found in `src/models/`.

5. **Model Evaluation**

   Evaluation metrics and validation procedures are in `src/models/`.

6. **Model Registration**

   Trained models are registered and versioned using MLflow.

7. **Pipeline Execution**

   The entire pipeline can be executed using DVC:

   ```bash
   dvc repro
   ```



## 🌐 Deployment

### Flask Application

1. **Navigate to Flask App Directory**

   ```bash
   cd flask_app
   ```



2. **Install Flask**

   ```bash
   pip install flask
   ```



3. **Run the Application**

   ```bash
   flask run
   ```



### Dockerization

1. **Build Docker Image**

   ```bash
   docker build -t fake-news-detection:latest .
   ```



2. **Run Docker Container**

   ```bash
   docker run -p 5000:5000 fake-news-detection:latest
   ```



### Kubernetes Deployment

1. **Create EKS Cluster**

   ```bash
   eksctl create cluster --name fake-news-cluster --region us-east-1 --nodegroup-name standard-workers --node-type t3.medium --nodes 3
   ```



2. **Deploy Application**

   ```bash
   kubectl apply -f deployment.yaml
   ```



3. **Access the Application**

   Retrieve the external IP:

   ```bash
   kubectl get svc fake-news-service
   ```



Access the application at `http://<external-ip>:5000`.

## 📊 Monitoring with Prometheus and Grafana

1. **Prometheus Setup**

   * Launch an EC2 instance.
   * Install Prometheus and configure it to scrape metrics from your application.

2. **Grafana Setup**

   * Launch another EC2 instance.
   * Install Grafana and configure it to use Prometheus as a data source.([arXiv][2])

3. **Create Dashboards**

   Set up dashboards in Grafana to monitor application metrics and performance.

## 🧪 Testing and CI/CD

* **Testing**

  Unit tests are located in the `tests/` directory. Run them using:

```bash
  pytest tests/
```



* **Continuous Integration**

  GitHub Actions workflows are defined in `.github/workflows/ci.yaml` to automate testing and deployment processes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
