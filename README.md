📊 End-to-End Telco Customer Churn Prediction (with MLflow & Docker)

This project is an end-to-end machine learning solution that predicts customer churn for a telecom company.
It’s not just about building a model — the goal is to turn predictions into business value by:

Tracking experiments with MLflow

Deploying everything inside Docker for reproducibility

Providing a Streamlit dashboard that shows live predictions and business KPIs

🎯 Goal

Build a reproducible ML platform that predicts churn and shows how much revenue a telecom company could save by retaining high-risk customers.

🌍 Vision

Enable telecom businesses to:

Reduce churn using data-driven insights

Monitor model performance over time with MLflow

Make better retention decisions with live KPIs


⚙️ Tech Highlights

🔄 End-to-End ML Pipeline: From data cleaning → training → evaluation → deployment

📈 MLflow: Logs parameters, metrics, and model artifacts for easy experiment tracking

🤖 Models: Logistic Regression (baseline) vs. Random Forest & XGBoost (ensemble methods)

🔧 Feature Engineering: Custom transformations to boost performance

🐳 Dockerized: Reproducible setup anywhere with a single command

💰 Business Impact: Predicts churn + estimates revenue saved if customers are retained

🚀 How to Run

Clone the Repo

git clone <repository_url>
cd churn-prediction-mlflow-docker


Start with Docker

docker-compose up -d


MLflow UI → http://localhost:5000

Streamlit App → http://localhost:8501

(Optional) Train the Model Locally

python src/train_ml.py


Saves the pipeline in models/final_churn_pipeline.pkl

📊 Results

✅ Model Performance: Final Random Forest model achieved ~98% test accuracy

💵 Business KPI:

Potential annual revenue saved: $1,030,694.16

Adjusted after 10% discount: $927,624.74

🏗️ Deployment: End-to-end workflow containerized for plug-and-play use

📸 Screenshots 

Streamlit UI - screenshots/streamlitui.png


📌 What’s Next

Experiment with deep learning models

Deploy to cloud (AWS/GCP/Azure) for production
