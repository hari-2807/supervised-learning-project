### Supervised Learning Model Comparison Pipeline
### Dataset: Pima Indians Diabetes Classification
##Models: Logistic Regression, Random Forest, XGBoost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

### Data Loading

print("="*60)
print("STEP 1: DATA LOADING")
print("="*60)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=columns)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print()

### Data Profiling

print("="*60)
print("STEP 2: DATA PROFILING")
print("="*60)
print("First 5 rows:")
print(df.head())
print()
print("Data types:")
print(df.dtypes)
print()
print("Missing values (shown as 0 in some medical fields):")
print((df == 0).sum())
print()
print("Descriptive statistics:")
print(df.describe())
print()
print("Target distribution:")
print(df['Outcome'].value_counts())
print(f"Diabetes positive rate: {df['Outcome'].mean():.2%}")
print()

### Feature Engineering

print("="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean = df.copy()
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Zero values imputed with median for medical columns")
print(f"Missing values after imputation: {df_clean.isnull().sum().sum()}")
print()

X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Features standardized with StandardScaler")
print()

### Model Training & Comparison

print("="*60)
print("STEP 4: MODEL TRAINING & COMPARISON")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

splits = [0.30, 0.20, 0.10]  
results = []

for test_size in splits:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTrain/Test Split: {1-test_size:.0%}/{test_size:.0%}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    for name, model in models.items():
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results.append({
            'Model': name,
            'Test_Size': f"{int(test_size*100)}%",
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4) if auc else None
        })

        print(f"  {name:20s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}" if auc else f"  {name:20s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

### Save Results

print()
print("="*60)
print("STEP 5: SAVE RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
results_df.to_csv('model_comparison_results.csv', index=False)
print("Results saved to: model_comparison_results.csv")

pickle.dump(scaler, open('scaler.pkl', 'wb'))
df_clean.to_csv('diabetes_data.csv', index=False)
print("Data and scaler saved")

for split in results_df['Test_Size'].unique():
    split_df = results_df[results_df['Test_Size'] == split]
    best_model_name = split_df.loc[split_df['F1_Score'].idxmax(), 'Model']
    print(f"Best model for {split} test split: {best_model_name}")

overall_best = results_df.groupby('Model')['F1_Score'].mean().idxmax()
print(f"\nOverall best model (avg F1 across splits): {overall_best}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
for name, model in models.items():
    model.fit(X_train, y_train)
    pickle.dump(model, open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb'))
print("All models saved as .pkl files")
print()
print("="*60)
print("PIPELINE COMPLETE")
print("="*60)
