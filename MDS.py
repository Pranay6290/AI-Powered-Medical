import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Title for the Streamlit app
st.title("🔬 AI-Powered Medical Diagnosis System")

# Sidebar for dataset selection
st.sidebar.title("Select Medical Dataset")
dataset_choice = st.sidebar.selectbox("Choose a dataset:", ["Diabetes",  "Breast Cancer"])

# Load dataset based on user selection
if dataset_choice == "Diabetes":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
elif dataset_choice == "Breast Cancer":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    columns = data.feature_names
    dataset = pd.DataFrame(X, columns=columns)
    dataset['Outcome'] = y
else:
    st.error("Invalid Dataset Selection")
    st.stop()

# Load dataset if not Breast Cancer
if dataset_choice != "Breast Cancer":
    try:
        dataset = pd.read_csv(url, names=columns) if columns else pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Try uploading the dataset manually.")
        st.stop()

# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(dataset.head())

# Data preprocessing
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar for model selection
st.sidebar.title("Choose Model")
model_type = st.sidebar.selectbox("Select the machine learning model", ["Random Forest", "Logistic Regression", "SVM"])

# Model selection
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_type == "Logistic Regression":
    model = LogisticRegression(random_state=42)
else:
    model = SVC(probability=True, random_state=42)

# Train model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.write(f"### {model_type} Accuracy: {accuracy * 100:.2f}%")

# ROC Curve and AUC Score
y_prob = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot ROC Curve
st.write(f"### {model_type} ROC Curve (AUC = {roc_auc:.4f})")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'{model_type} ROC Curve')
ax.legend()
st.pyplot(fig)

# User input section
st.write("### Enter Patient Data for Prediction")
input_data = []
for feature in X.columns:
    val = st.number_input(f"{feature}", min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))
    input_data.append(val)

# Making prediction
input_data_scaled = scaler.transform([input_data])
prediction = model.predict(input_data_scaled)

# Display prediction result
st.write("### Prediction Result")
if prediction[0] == 1:
    st.error("🔴 The patient is likely to have the condition.")
else:
    st.success("🟢 The patient is not likely to have the condition.")