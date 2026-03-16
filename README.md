# 📉 Customer Churn Prediction — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20CatBoost%20%7C%20RandomForest-orange?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

An **end-to-end Machine Learning web application** that predicts whether a telecom customer will churn (leave the service) or not — based on their usage patterns, contract type, and billing details.

The app takes **18 customer features** as input and returns a **churn prediction + probability score** in real time. Built with **Python + Flask**, trained using **XGBoost, CatBoost, and RandomForest** with **GridSearchCV** for hyperparameter tuning. Fully **Dockerized** for production-ready deployment.

---

## ✨ Features

- 🔮 **Real-time Churn Prediction** — Yes / No with probability %
- 📋 **18 Input Features** — contract, billing, services, tenure, charges
- 🤖 **Multiple ML Models** — XGBoost, CatBoost, RandomForest compared
- ⚙️ **GridSearchCV** — best hyperparameters auto-selected
- 🌐 **Flask Web UI** — clean form-based prediction interface
- 🐳 **Docker Support** — containerized, runs anywhere
- 📦 **Modular Pipeline** — separate train & predict pipelines
- 📝 **Logging** — all events tracked

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Backend | Python, Flask |
| ML Models | XGBoost, CatBoost, RandomForestClassifier |
| Model Selection | GridSearchCV (scikit-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | dill |
| Containerization | Docker |
| Packaging | setuptools |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py                        # Flask app — routes & prediction trigger
├── requirements.txt              # All dependencies
├── Dockerfile                    # Docker container config
├── setup.py                      # Package setup
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Load & split dataset
│   │   ├── data_transformation.py# Encoding, scaling, preprocessing
│   │   └── model_trainer.py      # Train & compare multiple models
│   │
│   ├── pipelines/
│   │   ├── train_pipeline.py     # End-to-end training pipeline
│   │   └── prediction.py         # CustomData class + PredictPipeline
│   │
│   ├── logger.py                 # Logging configuration
│   ├── exception.py              # Custom exception handler
│   └── utils.py                  # save_object, load_object, evaluate_models
│
├── templates/
│   ├── index.html                # Home page
│   └── predict.html              # Prediction form + results
│
├── artifacts/                    # Saved model, preprocessor files
├── notebook/                     # EDA & model experimentation
├── logs/                         # Application logs
└── catboost_info/                # CatBoost training logs
```

---

## 🧠 Input Features

| Feature | Type | Description |
|---|---|---|
| `SeniorCitizen` | Float | Is the customer a senior citizen? (0/1) |
| `Partner` | Categorical | Has a partner? (Yes/No) |
| `Dependents` | Categorical | Has dependents? (Yes/No) |
| `tenure` | Float | Months with the company |
| `PhoneService` | Categorical | Has phone service? |
| `MultipleLines` | Categorical | Multiple phone lines? |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Online security add-on? |
| `OnlineBackup` | Categorical | Online backup add-on? |
| `DeviceProtection` | Categorical | Device protection plan? |
| `TechSupport` | Categorical | Tech support add-on? |
| `StreamingTV` | Categorical | Streaming TV service? |
| `StreamingMovies` | Categorical | Streaming movies service? |
| `Contract` | Categorical | Month-to-month / 1yr / 2yr |
| `PaperlessBilling` | Categorical | Paperless billing? |
| `PaymentMethod` | Categorical | Credit card / Bank transfer / etc. |
| `MonthlyCharges` | Float | Monthly bill amount |
| `TotalCharges` | Float | Total amount charged |

---

## ⚙️ How It Works

```
Raw Input (18 features)
        ↓
data_transformation.py  →  Encoding + Scaling via preprocessor.pkl
        ↓
model_trainer.py  →  XGBoost / CatBoost / RandomForest + GridSearchCV
        ↓
Best Model saved  →  model.pkl
        ↓
PredictPipeline  →  Loads model.pkl + preprocessor.pkl
        ↓
Flask UI  →  Returns "Churn: Yes/No" + Probability %
```

---

## 🚀 Installation & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/angrajDEV/customer_churn_prediction.git
cd customer_churn_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🐳 Run with Docker

```bash
# Build the image
docker build -t customer-churn .

# Run the container
docker run -p 5000:5000 customer-churn
```

Open `http://localhost:5000` in your browser.

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| RandomForestClassifier | ~80%+ |
| XGBoostClassifier | ~80%+ |
| CatBoostClassifier | ~80%+ |

> Best model selected automatically via GridSearchCV and saved to `artifacts/model.pkl`

---

## 💡 Key Highlights

- 🔄 **Modular ML pipeline** — ingestion → transformation → training → prediction
- 🏆 **Multi-model comparison** — best model auto-selected via GridSearchCV
- 💾 **Artifact persistence** — model & preprocessor saved with `dill`
- 🐳 **Docker-ready** — zero environment issues on any machine
- 📝 **Custom logging & exception handling** — production-grade code structure
- 📓 **EDA Notebook** — exploratory analysis before modeling

---

## 👨‍💻 Author

**Angraj Dewangan (Nimmu)**  
MCA — Data Science & Machine Learning  
Guru Ghasidas University, Bilaspur

[![GitHub](https://img.shields.io/badge/GitHub-angrajDEV-black?style=flat&logo=github)](https://github.com/angrajDEV)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/)

---

> ⭐ If you found this project useful, do give it a star!
