import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from hpelm import ELM

# Title
st.title("CLOUD-ENABLED BREAST CANCER DIAGNOSTIC SYSTEM USING EXTREME LEARNING MACHINE")

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'elm' not in st.session_state:
    st.session_state.elm = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'y_proba' not in st.session_state:
    st.session_state.y_proba = None


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    features = data.feature_names
    target = data.target
    return df, features, target


def preprocess_and_split(df, features, target):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.2, random_state=42, stratify=target
    )
    return X_train, X_test, y_train, y_test, scaler


def train_elm(X_train, y_train, n_hidden=100):
    elm = ELM(X_train.shape[1], 1, classification="c")  # fix here: add output=1
    elm.add_neurons(n_hidden, "sigm")
    y_train_np = np.array(y_train).reshape(-1, 1)  # convert to numpy array with shape (n_samples, 1)
    elm.train(X_train, y_train_np)
    return elm


def evaluate_model(elm, X_test, y_test):
    y_test_np = np.array(y_test).reshape(-1, 1)  # convert to numpy array with shape (n_samples, 1)
    y_proba = elm.predict(X_test)
    y_pred = np.round(y_proba).astype(int).flatten()
    y_test_flat = y_test_np.flatten()  # flatten to 1D for metrics

    accuracy = accuracy_score(y_test_flat, y_pred)
    precision = precision_score(y_test_flat, y_pred)
    recall = recall_score(y_test_flat, y_pred)
    f1 = f1_score(y_test_flat, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test_flat, y_proba)
    roc_auc = auc(fpr, tpr)
    class_report = classification_report(y_test_flat, y_pred, output_dict=True)
    cm = confusion_matrix(y_test_flat, y_pred)

    return y_pred, y_proba, accuracy, precision, recall, f1, fpr, tpr, roc_auc, class_report, cm


# Step 1: Load Data
if st.button("Step 1: Load Wisconsin Breast Cancer Dataset"):
    st.session_state.df, st.session_state.features, st.session_state.target = load_data()
    st.success(f"Dataset loaded successfully with shape: {st.session_state.df.shape}")
    st.dataframe(st.session_state.df.head())
    st.session_state.data_loaded = True

# Step 2: Preprocess & Split
if st.session_state.data_loaded:
    if st.button("Step 2: Preprocess and Split Dataset"):
        X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
            st.session_state.df, st.session_state.features, st.session_state.target
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.success(f"Data preprocessed and split: Train size {X_train.shape[0]}, Test size {X_test.shape[0]}")
        st.write("Feature vector example (scaled):")
        st.write(pd.DataFrame(X_train, columns=st.session_state.features).head())

# Step 3: Train ELM Model
if st.session_state.X_train is not None:
    if st.button("Step 3: Train Extreme Learning Machine (ELM) Model"):
        elm = train_elm(st.session_state.X_train, st.session_state.y_train, n_hidden=100)
        st.session_state.elm = elm
        st.success("ELM model trained with 100 hidden neurons.")

# Step 4: Evaluate Model
if st.session_state.elm is not None:
    if st.button("Step 4: Evaluate Model on Test Data"):
        y_pred, y_proba, accuracy, precision, recall, f1, fpr, tpr, roc_auc, class_report, cm = evaluate_model(
            st.session_state.elm, st.session_state.X_test, st.session_state.y_test
        )
        st.session_state.y_pred = y_pred
        st.session_state.y_proba = y_proba

        st.subheader("Classification Report")
        st.text(classification_report(st.session_state.y_test, y_pred))

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"AUC: {roc_auc:.4f}")

        # Plot ROC Curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

        # Confusion matrix plot
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"])
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title('Confusion Matrix')
        st.pyplot(fig2)

# Step 5: Feature Distribution Visualization
if st.session_state.data_loaded:
    if st.button("Step 5: Show Feature Distribution"):
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=st.session_state.df[st.session_state.features[:6]].melt(var_name='Feature', value_name='Value'),
            x='Feature', y='Value', ax=ax3)
        ax3.set_title("Feature Distribution (First 6 Features)")
        st.pyplot(fig3)

# Step 6: Manual Input for Prediction
if st.session_state.scaler is not None and st.session_state.elm is not None:
    st.subheader("Step 6: Manual Input for Breast Cancer Prediction")
    user_inputs = {}
    with st.form(key='manual_input_form'):
        for feature in st.session_state.features:
            val = st.number_input(f"{feature}", value=0.0, format="%.4f")
            user_inputs[feature] = val
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        user_df = pd.DataFrame([user_inputs])
        user_scaled = st.session_state.scaler.transform(user_df)
        manual_proba = st.session_state.elm.predict(user_scaled)
        manual_pred = int(np.round(manual_proba).flatten()[0])
        classes = {0: "Malignant", 1: "Benign"}

        st.write(f"**Prediction:** {classes[manual_pred]}")
        st.write(f"**Probability score:** {manual_proba[0][0]:.4f}")
