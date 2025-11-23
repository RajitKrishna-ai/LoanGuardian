📌 LoanGuardian – Predictive Analytics for Loan Default & Automated Loan Processing
By Rajit R Krishna — Data Scientist | ML Engineer | Dubai, UAE

🚀 1. Project Overview
LoanGuardian is an end-to-end Machine Learning solution designed to help banks in the UAE identify potential loan defaulters early and automate loan processing across the stages:
Lead → File → Sanction → Disbursement
The project includes:
✔ Synthetic UAE banking dataset
✔ Exploratory Data Analysis (EDA)
✔ Feature Engineering
✔ Machine Learning Models
✔ XGBoost-based risk scoring
✔ Airflow automation pipelines
✔ Flask API for real-time prediction
✔ Explainable, auditor-friendly ML workflow


This project follows the strict explainability and transparency requirements commonly requested by UAE financial institutions.

👤 2. About Me
My name is Rajit R Krishna based in Dubai, UAE.
Originally from Kerala, raised in Delhi, and graduated in Aeronautical Engineering (2018).
I worked 3+ years at Xoriant in India as a Software Engineer (Data Scientist), where I delivered ML/NLP/GenAI solutions for banking clients.
This project represents my ability to design an industry-grade, production-ready ML system following UAE banking standards.

🏦 3. Problem Statement
✔ A leading UAE bank struggled with:
✔ Slow loan processing between stages
✔ High default rates due to late risk detection
✔ Manual underwriting decisions
✔ Inconsistent borrower profiling
✔ Difficulty in identifying high-risk borrowers early



🎯 4. Project Goals
LoanGuardian was built to:
✔ Predict potential defaulters early in the pipeline
✔ Reduce loan processing time (TAT)
✔ Automate ML scoring + workflows
✔ Improve recall (catch more defaulters)
✔ Provide explainable ML insights for risk teams

📊 5. Business Impact (Synthetic but Realistic)
MetricImpactDefault prediction accuracy 91%
Improvement in recall                  +18%
Manual approval time reduction          30%
TAT reduction in processing             20%
Improved loan approval accuracy         	15%

🗂 6. Repository Structure
LoanGuardian/
│
├── data/                    # Synthetic UAE loan dataset
├── notebooks/               # EDA, Feature Engineering, Training
├── src/                     # ML pipeline modules
├── pipeline/                # Airflow DAGs
├── deployment/
│     ├── model/             # Saved trained models
│     └── api/               # Flask inference API
├── docs/                    # Architecture diagrams, flowcharts
└── README.md


📁 7. Synthetic UAE Dataset
The dataset simulates realistic UAE borrower behavior with columns such as:
✔ LoanAmount_AED
✔ MonthlyIncome_AED
✔ Age
✔ Emirate (Dubai, Abu Dhabi, Sharjah, Ajman)
✔ EmploymentType (Salaried, Self-Employed, Business Owner)
✔ LoanType (Personal, Auto, Credit Card, SME Loan)
✔ Nationality
✔ Dependents
✔ CreditScore
✔ RepaymentHistoryScore
✔ LoanTenureMonths
✔ DefaultStatus (0/1)
✔ Dataset size: 10,000 rows

📘 8. Exploratory Data Analysis (EDA)
Includes:
✔ Missing value analysis
✔ Outlier detection
✔ Univariate analysis
✔ Bivariate analysis
✔ Loan Amount vs Income
✔ Credit Score distributions
✔ Correlation heatmap
✔ Emirate-wise borrower behavior
✔ Screenshots included in docs/eda/*.png

🛠 9. Feature Engineering
Steps applied:
✔ Missing Value Imputation
✔ Outlier Removal (IQR method)
✔ One-Hot Encoding
✔ Label Encoding
✔ WOE Encoding
✔ Scaling (MinMax + Standard)
✔ Feature Selection:
   Information Value (IV)
   Variance Inflation Factor (VIF)
   Correlation Filtering





🤖 10. Machine Learning Models
The following models were trained + evaluated:
✔ Logistic Regression
✔ Decision Tree
✔ Random Forest
✔ XGBoost (Best Model)

Evaluation:
✔ Accuracy
✔ Recall
✔ Precision
✔ F1 Score
✔ ROC-AUC
✔ Confusion matrix


✔ Final chosen model: XGBoost (91% Accuracy)

⚙️ 11. Airflow Automation Workflow
Airflow DAGs automate:
✔ Data Ingestion
✔ Preprocessing
✔ Feature Engineering
✔ Model Training
✔ Daily Predictions
✔ Storing Scores in a Database

DAGs located in: pipeline/airflow_dags

🌐 12. Flask API (Real-Time Prediction)
Endpoints:
Endpoint     Purpose
/predict     Predicts default risk
/retrain     Retrains model using new data
/health      API health status
API path: deployment/api/app.py


🧩 13. System Architecture Diagram

┌────────────┐
│   Frontend │
└─────┬──────┘
      │
┌─────▼──────┐
│  Flask API │
└─────┬──────┘
      │
┌─────▼──────────┐
│   ML Model     │
└─────┬──────────┘
      │
┌─────▼───────────┐
│   Feature Store │
└─────────────────┘


🧪 14. My Contribution (End-to-End)
I completed:
✔ Data generation (synthetic UAE banking dataset)
✔ EDA + Feature Engineering
✔ ML model training & tuning
✔ End-to-end architecture design
✔ Airflow automation
✔ Flask API deployment
✔ Documentation + reporting
This reflects real-world ML engineering + data science workflow.

📌 15. How to Run Locally
pip install -r requirements.txt
cd deployment/api
python app.py


🤝 16. Contact
Rajit R Krishna
Data Scientist | ML Engineer
Dubai, UAE
rajitkrishna94@gmail.com
