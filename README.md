<div align="center">

# 🩺 Supervised Learning Model Comparison

**Use Case: Diabetes Prediction | Classification Benchmark**

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Models Compared](#-models-compared)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

This project builds an end-to-end machine learning pipeline to predict diabetes onset using the **Pima Indians Diabetes Dataset**. Three supervised learning models were trained, evaluated, and compared across multiple train/test splits to identify the best-performing model for this healthcare classification task.

The workflow covers:
- Data loading and profiling
- Data cleaning and feature engineering
- Model training and evaluation
- Model comparison across different train/test splits
- Interactive prediction using a Streamlit web app

---

## ✨ Features

- **Data Preprocessing Pipeline** — Handles zero-value imputation in medical columns and applies feature scaling
- **Multi-Model Evaluation** — Compares Logistic Regression, Random Forest, and XGBoost
- **Interactive Dashboard** — Built with Streamlit for model comparison and prediction
- **Feature Importance Analysis** — Shows the most influential features for prediction
- **ROC Curve & Confusion Matrix** — Visual evaluation for classification performance
- **Robustness Testing** — Tests performance across multiple train/test splits

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Core Libraries** | pandas, numpy, scikit-learn, xgboost |
| **Visualization** | matplotlib, seaborn, plotly |
| **App Framework** | Streamlit |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```bash
supervised_learning_project/
├── data_analysis.py              # Data profiling, preprocessing, model training, evaluation
├── app.py                        # Streamlit interactive application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── model_comparison_results.csv  # Exported metrics across all splits
├── scaler.pkl                    # Saved StandardScaler object
├── logistic_regression_model.pkl # Saved Logistic Regression model
├── random_forest_model.pkl       # Saved Random Forest model
├── xgboost_model.pkl             # Saved XGBoost model
└── diabetes_data.csv             # Processed dataset
```

---

## 📊 Dataset

- **Source:** Pima Indians Diabetes Dataset (UCI Machine Learning Repository)
- **Records:** 768 patient records
- **Features:** 8 input features
- **Target:** `Outcome`  
  - `1` = Diabetes Positive
  - `0` = Diabetes Negative

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |

---

## 🤖 Models Compared

| Model | Type | Best For |
|-------|------|----------|
| **Logistic Regression** | Linear Model | Interpretability, simple baseline, probability-based output |
| **Random Forest** | Bagging Ensemble | Strong default performance, feature importance, reduced overfitting |
| **XGBoost** | Boosting Ensemble | High predictive power, tabular datasets, model benchmarking |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/supervised_learning_project.git ## Replace YOUR_USERNAME with actual username
cd supervised_learning_project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### 1. Run the data analysis pipeline

```bash
python data_analysis.py
```

This script will:
- Load the dataset
- Profile and clean the data
- Handle missing/invalid zero values
- Scale the features
- Train all three models
- Evaluate them on different train/test splits
- Save the trained models and results

### 2. Launch the Streamlit application

```bash
streamlit run app.py
```

The app includes:
- **Data Overview** — Dataset preview and distributions
- **Model Comparison** — Side-by-side metric comparison
- **Detailed Metrics** — Confusion matrix, ROC curve, and feature importance
- **Prediction** — Enter patient values and get a diabetes prediction

---

## 📈 Results

The models were evaluated across three train/test splits: **70/30**, **80/20**, and **90/10**. Performance was measured using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

### Performance Summary

| Train/Test Split | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|-------|----------|-----------|--------|----------|---------|
| 70/30 | Logistic Regression | 0.745 | 0.672 | 0.531 | 0.593 | 0.836 |
| 70/30 | Random Forest | 0.745 | 0.672 | 0.531 | 0.593 | 0.827 |
| 70/30 | XGBoost | 0.745 | 0.645 | 0.605 | 0.624 | 0.794 |
| 80/20 | Logistic Regression | 0.701 | 0.587 | 0.500 | 0.540 | 0.813 |
| 80/20 | Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.817 |
| 80/20 | XGBoost | 0.753 | 0.660 | 0.611 | 0.635 | 0.804 |
| 90/10 | Logistic Regression | 0.727 | 0.615 | 0.593 | 0.604 | 0.834 |
| 90/10 | Random Forest | 0.779 | 0.692 | 0.667 | 0.679 | 0.874 |
| 90/10 | XGBoost | 0.805 | 0.714 | 0.741 | 0.727 | 0.868 |

### Best Model by Split

| Split | Best Model | Why It Performed Best |
|-------|------------|-----------------------|
| 70/30 | XGBoost | Highest F1-score (0.624) and better recall than the other two models |
| 80/20 | XGBoost | Best balance of accuracy, precision, recall, and F1-score |
| 90/10 | XGBoost | Highest accuracy (0.805) and highest F1-score (0.727) |

### Model Ranking

1. **XGBoost** — Best overall model across all train/test splits  
2. **Random Forest** — Strong second-best model, especially on the 90/10 split  
3. **Logistic Regression** — Good interpretable baseline, but lower predictive performance  

### Final Recommendation

For this diabetes prediction use case, **XGBoost** is the recommended model because it delivered the most consistent and strongest classification performance across all train/test splits.

- Choose **XGBoost** when predictive performance is the main priority
- Choose **Random Forest** when you want strong performance with easier interpretability
- Choose **Logistic Regression** when you need a simple and explainable baseline model

---

## 💡 Key Insights

- **Feature engineering mattered:** Zero values in medical columns such as Glucose, BloodPressure, Insulin, SkinThickness, and BMI were treated as invalid values and replaced using median imputation.
- **F1-score was the key metric:** Since this is a healthcare classification problem, F1-score is more useful than accuracy alone because it balances precision and recall.
- **XGBoost was the most consistent model:** It achieved the best F1-score across all three train/test splits.
- **Random Forest was a strong alternative:** It performed especially well on ROC-AUC and offers useful feature importance.
- **Logistic Regression remained valuable as a baseline:** It is easier to explain and useful in scenarios where interpretability matters more than raw performance.

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Pima Indians Diabetes](https://archive.ics.uci.edu/ml/datasets/diabetes)
- Built as a supervised learning portfolio project
- Mentorship and guidance by [**Harshit Tripathi**](https://github.com/harshitboots)

---

<p align="center">Made with ❤️ by Hari</p>