# 📊 Customer Churn Prediction

An end-to-end data science project that predicts whether a customer is likely to churn,
identifies the key drivers of churn, and turns those insights into actionable retention
strategies — complete with EDA, multiple ML models, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧭 Project Overview

Customer churn — when a customer stops doing business with a company — is one of the
costliest problems in subscription and service industries. Acquiring a new customer can
cost **5–25x more** than retaining an existing one, so even small reductions in churn
meaningfully improve revenue and customer lifetime value.

This project builds a complete churn-prediction pipeline:

1. **Data cleaning & preprocessing** of a realistic telecom customer dataset
2. **Exploratory Data Analysis (EDA)** uncovering the behavioral and contractual drivers of churn
3. **Feature engineering** to create business-meaningful predictive signals
4. **Five machine learning models** trained, tuned, and compared head-to-head
5. **An interactive Streamlit dashboard** for real-time churn scoring and business reporting

> 📌 **Note on the dataset:** `data/customer_churn.csv` is synthetically generated
> (`data/generate_dataset.py`) to mirror the structure and statistical relationships of
> well-known public telecom churn datasets (e.g., IBM Telco Customer Churn), avoiding any
> licensing restrictions while preserving realistic, learnable churn signal.

---

## ✨ Features

- **Full ML pipeline** — cleaning, encoding, scaling, outlier handling, train/test split
- **9+ EDA visualizations** — churn distribution, contract/payment/tenure analysis, correlation heatmap, customer segmentation
- **5 ML models compared** — Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM
- **Full evaluation suite** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrices, ROC Curves
- **Feature importance analysis** to explain *why* the model predicts what it predicts
- **Interactive Streamlit dashboard** with:
  - KPI cards (total customers, churn rate, revenue at risk, etc.)
  - Interactive Plotly charts with live filtering
  - A real-time churn prediction form with a visual risk gauge
  - Model performance comparison view
  - Business insights & retention recommendations
  - CSV export of filtered data and individual predictions
  - Light/Dark theme toggle
- **Customer segmentation** via KMeans clustering on tenure, charges, and support calls
- **Production-style code** — modular, commented, cached, and ready to deploy

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn, XGBoost |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── generate_dataset.py            # Synthetic dataset generator
│   ├── customer_churn.csv             # Raw dataset (with realistic messiness)
│   └── customer_churn_cleaned.csv     # Cleaned + feature-engineered dataset
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb  # Full EDA -> modeling -> evaluation notebook
│
├── models/
│   ├── customer_churn_model.pkl       # Best trained model (also copied to project root)
│   ├── scaler.pkl                     # StandardScaler fitted on training data
│   ├── feature_columns.pkl            # Exact feature column order used in training
│   ├── numeric_features.pkl           # Columns the scaler applies to
│   └── model_metadata.json            # Metrics, feature importances, best model name
│
├── dashboard/
│   └── app.py                         # Streamlit dashboard implementation
│
├── images/                            # All generated EDA & model evaluation charts
│
├── reports/
│   └── Customer_Churn_Business_Report.md  # Business-facing findings & recommendations
│
├── app.py                             # Root entry point (`streamlit run app.py`)
├── requirements.txt
├── README.md
└── customer_churn_model.pkl           # Copy of the best model at the project root
```

---

## ⚙️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

### Regenerate the dataset (optional — a copy is already included)
```bash
cd data
python generate_dataset.py
```

### Run the full analysis & training notebook
```bash
jupyter notebook notebooks/Customer_Churn_Prediction.ipynb
```
This notebook performs data cleaning, EDA, feature engineering, trains all 5 models,
evaluates them, and saves the best model + metadata into `models/`.

### Launch the interactive dashboard
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (typically `http://localhost:8501`) in your browser.

**Dashboard tabs:**
- **📈 Churn Overview** — KPI cards, churn distribution, contract/payment/tenure breakdowns, segmentation
- **🔮 Predict Churn** — fill in a customer profile and get an instant churn risk score + gauge + recommended action
- **🧪 Model Performance** — compare all 5 models on every metric, view feature importance
- **💡 Business Insights** — churn drivers, high-risk profile, retention strategy, downloadable report

---

## 📊 Results and Insights

- **Best model:** Logistic Regression — **ROC-AUC 0.79**, **Recall 0.74** on the churn class
- **Top churn drivers:** Tenure, Total/Monthly Charges, Contract Type (Month-to-Month vs.
  Two Year), Customer Support Calls, Internet Service type
- **Churn is concentrated** in customers with under 12 months of tenure (~54% churn rate)
  vs. customers with 49+ months of tenure (~14% churn rate)
- **Month-to-month contracts churn at 43%**, more than double the rate of two-year contracts (17.5%)
- **4+ support calls is a strong early-warning signal** — churn rate rises sharply beyond this point

Full breakdown and business recommendations are in
[`reports/Customer_Churn_Business_Report.md`](reports/Customer_Churn_Business_Report.md).

---

## 🔮 Future Enhancements

- [ ] Retrain on real, company-specific historical churn data
- [ ] Add SHAP-based explainability for individual predictions in the dashboard
- [ ] Hyperparameter tuning via GridSearchCV / Optuna for further model improvement
- [ ] Add customer satisfaction survey scores and usage/data-consumption metrics
- [ ] Build an automated weekly scoring pipeline with email/Slack alerts for high-risk customers
- [ ] A/B testing framework to measure the causal impact of retention offers
- [ ] User authentication for the dashboard in a production deployment

---

## ☁️ Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full step-by-step instructions on:
- Pushing this project to GitHub
- Deploying the dashboard on Streamlit Community Cloud

---

## 📄 License

This project is released under the MIT License — free to use for learning, portfolio, and
internship/academic submission purposes.

---

## 🙋 About This Project

This project was built as a complete, portfolio-ready Data Analyst / Data Scientist
project, suitable for internship applications, academic submissions, and as a template
for real-world churn prediction systems. It demonstrates the full data science lifecycle:
business understanding → data cleaning → EDA → feature engineering → modeling →
evaluation → deployment-ready dashboard → business storytelling.
