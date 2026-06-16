# Supervised Learning Model Comparison

## Use Case: Diabetes Prediction
This project compares three supervised learning models on the Pima Indians Diabetes dataset to predict whether a patient has diabetes based on diagnostic measurements.

## Models Compared
1. **Logistic Regression** - Linear baseline, highly interpretable
2. **Random Forest** - Ensemble method, handles non-linearity well
3. **Gradient Boosting (XGBoost)** - Sequential ensemble, often best performance

## Repository Structure
- `data_analysis.py` - Data profiling, feature engineering, model training
- `app.py` - Streamlit application for interactive model comparison
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run analysis: `python data_analysis.py`
3. Launch app: `streamlit run app.py`

## Key Findings
- Model comparison across different train/test splits
- Feature importance analysis
- Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
