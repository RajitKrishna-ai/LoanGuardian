# **LoanGuardian – AI-Powered Loan Default Prediction & Automation System**

LoanGuardian is an **end-to-end AI system** designed to help banks and financial institutions **predict loan defaults early** 
and **automate loan processing workflows**.
It combines machine learning, workflow automation, and production-ready deployment to simulate **enterprise-level credit risk management**.

> 🚀 Built and modified for UAE financial institutions, optimized for real-world banking workflows, and designed for production-ready AI solutions.

---

## 🌟 **Key Features**

### 💰 **1. Predictive Loan Default Modeling**

* Trains advanced ML models: **Logistic Regression, Random Forest, XGBoost**
* Handles imbalanced datasets with **SMOTE** and class weighting
* Produces interpretable **risk scores** for informed decision-making

### ⚙️ **2. Automated ML Pipelines (Airflow)**

* Scheduled **data ingestion, preprocessing, and feature engineering**
* Automated **model training and evaluation**
* Alerts for **model drift** or performance degradation
* Enterprise-grade workflow orchestration

### 🌐 **3. Real-Time Prediction API (Flask)**

* REST API for instant loan default predictions
* Accepts JSON input: salary, loan amount, credit score, repayment history
* Returns **risk probability** and **confidence score**
* Ready for integration with bank dashboards or CRM systems

### 📊 **4. Feature Engineering & Insights**

* Exploratory Data Analysis (EDA) to identify trends and correlations
* Outlier detection and risk segmentation
* Optimized feature transformations for maximum model accuracy

### 🔍 **5. Model Explainability**

* SHAP-based **feature importance visualization**
* Clear explanation of predictions for compliance and audit
* Enables risk analysts to interpret model decisions

---

## 🛠️ **Tech Stack**

| Category            | Tools                            |
| ------------------- | ------------------------------- |
| Machine Learning    | Scikit-Learn, XGBoost, LightGBM |
| Data Handling       | Pandas, NumPy                   |
| Visualization       | Matplotlib, Seaborn, SHAP        |
| Workflow Automation | Apache Airflow                  |
| Deployment          | Flask                           |
| Environment         | Python, Jupyter Notebook        |

---

## 📁 **Project Structure**

```
LoanGuardian/
│
├── airflow/      # Automated DAGs for pipelines
├── data/         # Synthetic AED-based loan dataset
├── deployment/   # Flask API for real-time predictions
├── notebooks/     # EDA, Feature Engineering, Model Training
├── docs/         # Reports, visualizations, and metrics
└── README.md
```

---

## 🔄 **Workflow**

```
Raw Loan Data
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Model Training & Evaluation
      ↓
Airflow Pipeline Automation
      ↓
Flask API for Real-Time Predictions
      ↓
Risk Scores & SHAP-Based Explainability
```

---

## 📊 **Sample Output**

| Feature         | Example     |
| --------------- | ----------- |
| Loan Amount     | 150,000 AED |
| Salary          | 18,000 AED  |
| Credit Score    | 720         |
| Tenure          | 36 months   |
| Late Payments   | 1           |
| Employment Type | Private     |
| Loan Type       | Personal    |

**API Response**

```json
{
    "loan_default_risk": 1,
    "confidence": 0.87
}
```

---

## 🎯 **Benefits**

* Early identification of high-risk loans
* Automates repetitive banking workflows
* Transparent and interpretable risk scoring
* Production-ready and scalable for UAE banks
* Supports data-driven credit decisions
