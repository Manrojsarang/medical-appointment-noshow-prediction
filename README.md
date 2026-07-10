#  Medical Appointment No-Show Prediction


[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://medical-appointment-noshow-prediction.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/Manrojsarang/medical-appointment-noshow-prediction)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-blue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-blue?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)
 
A machine learning web application that predicts whether a patient is likely to miss a scheduled medical appointment using a tuned Random Forest classifier.

**Live Demo:** https://medical-appointment-noshow-prediction.streamlit.app/

**GitHub Repository:** https://github.com/Manrojsarang/medical-appointment-noshow-prediction

##  Table of Contents

- Project Overview
- Problem Statement
- Dataset
- Technologies Used
- Machine Learning Workflow
- Model Performance
- Project Structure
- Application Screenshots
- Installation
- Future Improvements
- Author


##  Project Overview

Missed medical appointments (No-Shows) create challenges for healthcare providers by increasing waiting times, wasting medical resources, and reducing operational efficiency. This project uses Machine Learning to predict whether a patient is likely to miss a scheduled appointment based on demographic, medical, and appointment-related information.

A **Random Forest Classifier** was trained, evaluated, and deployed through a **Streamlit web application**, allowing users to enter patient information and receive real-time predictions.

---

##  Problem Statement

Healthcare providers frequently face appointment no-shows, which lead to:

1. Wasted appointment slots
2. Increased healthcare costs
3. Longer waiting lists
4. Reduced hospital efficiency

The objective of this project is to predict whether a patient will attend or miss a scheduled appointment so hospitals can take preventive actions such as sending reminders or rescheduling appointments.

---

##  Dataset

The project uses the **Medical Appointment No-Show Dataset**, which contains appointment records collected from public healthcare facilities.

| Attribute       | Details                                            |
| --------------- | -------------------------------------------------- |
| Dataset Size    | **110,527** patient appointment records            |
| Problem Type    | Binary Classification                              |
| Target Variable | **No-Show** (0 = Attended, 1 = Missed Appointment) |

### Features Used

* Gender
* Age
* Scholarship
* Hypertension
* Diabetes
* Alcoholism
* Handicap
* SMS Received
* Waiting Days
* Scheduled Hour
* Appointment Weekday
* Neighbourhood

The objective is to predict whether a patient is likely to **attend** or **miss** a scheduled medical appointment based on these features.


**Target Variable**

* No-Show (Attend / Miss Appointment)

---

##  Machine Learning Workflow

The project follows a complete end-to-end Machine Learning pipeline:

1. Data Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Data Preprocessing
7. Model Training
8. Hyperparameter Tuning using **GridSearchCV**
9. Model Evaluation
10. Feature Importance Analysis
11. Model Serialization using **Joblib**
12. Streamlit Web Application Development
13. Version Control using Git & GitHub

---
##  Model Performance

The final model selected for deployment is a **Random Forest Classifier** optimized using **GridSearchCV**.

### Evaluation Metrics

| Metric   |      Value |
| -------- | ---------: |
| Accuracy | **79.74%** |

### Why Random Forest?

* Handles both numerical and categorical features effectively.
* Reduces overfitting through ensemble learning.
* Provides feature importance scores for model interpretability.
* Delivers robust performance on structured tabular datasets.

### Most Important Features

The trained model identified the following features as the most influential:

1. Waiting Days
2. SMS Received
3. Age
4. Scheduled Hour
5. Hypertension

Feature importance is also visualized in the Streamlit application.


##  Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model | Random Forest Classifier |
| Web Framework | Streamlit |
| Development | JupyterLab, VS Code |
| Version Control | Git, GitHub |

---

##  Machine Learning Model

Several classification algorithms were explored, and the final deployed model is:

**Random Forest Classifier (Hyperparameter Tuned using GridSearchCV)**

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Special attention was given to precision and recall because the dataset is imbalanced, making accuracy alone insufficient for evaluating performance.

---

## Deployment

The trained model was saved using **Joblib** and deployed with **Streamlit**.

The application allows users to:

* Enter patient information
* Predict appointment attendance
* View prediction probabilities
* Display feature importance
* Provide an interactive user interface for real-time predictions

---

##  Project Structure

```text
medical-appointment-noshow-prediction/
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── assets/
│   ├── home.png
│   ├── attend_prediction.png
│   ├── no_show_prediction.png
│   └── feature_importance.png
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── cleaned_medical_appointments.csv
│
├── models/
│   ├── medical_no_show_model.pkl
│   ├── feature_columns.pkl
│   └── feature_importance.csv
│
├── notebooks/
│   ├── 1.data_understanding.ipynb
│   ├── 2.data_cleaning_preprocessing.ipynb
│   ├── 3.EDA.ipynb
│   ├── 4.Model_Training.ipynb
│   └── 5.Prediction.ipynb
│
├── src/
│   └── preprocessing.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```



---

##  Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Manrojsarang/medical-appointment-noshow-prediction.git
```

### 2. Navigate to the project directory

```bash
cd medical-appointment-noshow-prediction
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open automatically in your default web browser.

---

## 🚀 Future Improvements

* Improve recall for minority class predictions.
* Compare performance with **XGBoost**, **LightGBM**, and **CatBoost**.
* Build a REST API using **FastAPI**.
* Deploy the application on a cloud platform.
* Add model monitoring and logging.
* Integrate a database for storing appointment records.
* Implement automated retraining with new data.

---

## Application Screenshots

### Home Page
![Home](https://raw.githubusercontent.com/Manrojsarang/medical-appointment-noshow-prediction/main/assets/home.png)

### Attend Prediction
![Attend Prediction](./assets/attend_prediction.png)

### No-Show Prediction
![No-Show Prediction](./assets/no_show_prediction.png)

### Feature Importance
![Feature Importance](./assets/feature_importance.png)

##  Author

**Manrojpreet**

This project was developed as an end-to-end Machine Learning solution for predicting medical appointment no-shows using a tuned Random Forest Classifier. It demonstrates the complete ML workflow, from data preprocessing and exploratory data analysis to model training, evaluation, deployment with Streamlit, and version control using Git and GitHub.
