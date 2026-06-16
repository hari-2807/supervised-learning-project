import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Supervised Learning Model Comparison",
    page_icon="",
    layout="wide"
)

# Title
st.title("Supervised Learning Model Comparison")
st.subheader("Use Case: Diabetes Prediction")

st.markdown("""
This application compares three supervised learning models on the Pima Indians Diabetes dataset:
- **Logistic Regression** — Linear baseline, highly interpretable, good for understanding feature relationships
- **Random Forest** — Ensemble of decision trees, handles non-linearity, provides feature importance
- **XGBoost** — Gradient boosting, sequential error correction, often highest predictive performance
""")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Configuration")

test_size = st.sidebar.selectbox(
    "Select Test Size",
    ["20% (Train 80%)", "30% (Train 70%)", "10% (Train 90%)"],
    index=0
)

test_size_map = {"20% (Train 80%)": 0.2, "30% (Train 70%)": 0.3, "10% (Train 90%)": 0.1}
test_size_val = test_size_map[test_size]

selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    ["Logistic Regression", "Random Forest", "XGBoost"],
    default=["Logistic Regression", "Random Forest", "XGBoost"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About the Models")

model_info = {
    "Logistic Regression": "Linear model. Best for: baseline comparison, probabilistic interpretation, when interpretability is critical.",
    "Random Forest": "Tree ensemble. Best for: handling mixed data types, feature importance analysis, reducing overfitting through averaging.",
    "XGBoost": "Gradient boosting. Best for: maximum predictive accuracy, structured/tabular data, competitive performance."
}

for m in selected_models:
    st.sidebar.markdown(f"**{m}**")
    st.sidebar.caption(model_info[m])

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Model Comparison", "Detailed Metrics", "Prediction"])

# ===== TAB 1: Data Overview =====
with tab1:
    st.header("Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(df))
        st.metric("Features", len(df.columns)-1)
    with col2:
        st.metric("Diabetes Positive", df['Outcome'].sum())
        st.metric("Diabetes Negative", len(df) - df['Outcome'].sum())

    st.subheader("Sample Data")
    st.dataframe(df.head(10), width='stretch')

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), width='stretch')

    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue='Outcome', kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature} by Outcome")
    st.pyplot(fig)

# ===== TAB 2: Model Comparison =====
with tab2:
    st.header("Model Comparison")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb

    # Data preprocessing
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size_val, random_state=42, stratify=y
    )

    st.write(f"Train/Test Split: {len(X_train)} / {len(X_test)} samples")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = []

    for name in selected_models:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': auc
        })

    results_df = pd.DataFrame(results)

    st.subheader("Performance Metrics")
    st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], color='green'), 
                 width='stretch')

    # Bar chart comparison
    fig = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                 x='Model', y='Score', color='Metric', barmode='group',
                 title="Model Performance Comparison",
                 labels={'Score': 'Score', 'Model': 'Model'},
                 height=500)
    st.plotly_chart(fig, width='stretch')

    # Best model recommendation
    best_model = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
    st.success(f"Best performing model for this split: **{best_model}** (based on F1 Score)")

    st.markdown("""
    **Model Selection Guidance:**
    - **Logistic Regression** is preferred when you need interpretability and probabilistic outputs.
    - **Random Forest** is preferred when you want robust performance with less tuning and need feature importance.
    - **XGBoost** is preferred when maximum predictive accuracy is the goal and you can handle more complexity.
    """)

# ===== TAB 3: Detailed Metrics =====
with tab3:
    st.header("Detailed Analysis")

    if len(selected_models) > 0:
        selected_model = st.selectbox("Select Model for Deep Dive", selected_models)

        model = models[selected_model]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{selected_model} - Confusion Matrix')
            st.pyplot(fig)

        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            fig.update_layout(
                title=f'{selected_model} - ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, width='stretch')

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                         title=f"{selected_model} - Feature Importance",
                         height=400)
            st.plotly_chart(fig, width='stretch')
        elif hasattr(model, 'coef_'):
            st.subheader("Feature Coefficients")
            coef = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)

            fig = px.bar(coef, x='Coefficient', y='Feature', orientation='h',
                         title=f"{selected_model} - Feature Coefficients",
                         height=400)
            st.plotly_chart(fig, width='stretch')

# ===== TAB 4: Prediction =====
with tab4:
    st.header("Make a Prediction")
    st.markdown("Enter patient data to predict diabetes risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        bp = st.number_input("Blood Pressure", 0, 150, 70)

    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col3:
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 100, 30)

    pred_model = st.selectbox("Select Model for Prediction", selected_models)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        model = models[pred_model]
        model.fit(X_train, y_train)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error(f"Prediction: **Diabetes Positive** (Confidence: {probability[1]:.1%})")
        else:
            st.success(f"Prediction: **Diabetes Negative** (Confidence: {probability[0]:.1%})")

        st.info(f"Model used: {pred_model}")
