# 🏦 Loan Eligibility Prediction App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📌 Project Overview
This project is a machine learning-based web application that predicts whether a loan applicant is likely to repay their loan. It aims to assist financial institutions in minimizing default risks by automating the initial eligibility assessment.

The application takes applicant details (Age, Income, Credit Score, etc.) via a user-friendly web interface, processes the data through a machine learning pipeline, and returns a real-time prediction with a probability score.

---

## 🚀 Key Features
* **End-to-End Deployment:** Full integration of a Machine Learning model with a Flask web backend.
* **Robust Preprocessing:** Implements `Log Transformation` for skewed income data and `StandardScaler` to normalize features, ensuring the model receives data in the exact format it was trained on.
* **Real-time Prediction:** Uses a serialized (Pickle) model to generate instant results.
* **Data Integrity:** Validates user inputs to prevent model crashes.

---

## 🛠️ Tech Stack
* **Frontend:** HTML, CSS (Jinja2 Templates)
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Model:** Linear Regression (Chosen for interpretability and baseline performance)

---

## 🧠 Model & Approach

### Why Linear Regression?
I chose Linear Regression as the baseline model to prioritize **interpretability**. In financial contexts, understanding *how* factors like 'Credit Score' or 'DTI' (Debt-to-Income Ratio) impact the decision is crucial.

### Feature Engineering Highlights
1.  **Log Transformation:** Applied to `Annual Income` to handle right-skewed distributions and reduce the impact of outliers.
2.  **Scaling:** Used `StandardScaler` to bring all features (e.g., Income vs. Age) to a comparable scale/unit.

---

## 📊 Performance
* **Accuracy:** ~88%
* **Focus:** Emphasized minimizing False Negatives (predicting a defaulter as a payer) to protect financial liability.

---

## 📂 Project Structure
```text
├── app.py               # Main Flask application
├── model.pkl            # Serialized Machine Learning Model
├── scaler.pkl           # Serialized Scaler (for consistent preprocessing)
├── requirements.txt     # List of dependencies
├── templates/
│   └── index.html       # Web Interface (HTML)
├── static/
│   └── style.css        # Styling (CSS)
└── README.md            # Project Documentation