import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load both models from Pickle
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model_intensity = load_model('brake_intensity_model.pkl')
model_duration = load_model('brake_duration_model.pkl')

# Display logo at top
st.image("logo.jpg", use_container_width=True)

st.title("🧠 Random Forest Model Dashboard")
st.markdown("This dashboard evaluates and predicts **Brake Intensity** (multi-class) and **Brake Duration** (binary classification) using Random Forest models.")

# Sidebar selection for model type
st.sidebar.header("🔧 Select Model")
selected_model = st.sidebar.selectbox("Choose the target to evaluate and predict:", ("Brake Intensity", "Brake Duration"))

# Sidebar option for choosing data source
data_option = st.sidebar.radio("Select Data Source:", ("Use Sample Test Data", "Upload Your Own Data"))

# Sidebar download sample CSVs
st.sidebar.header("📥 Download Sample CSV")

# Load real samples
sample_intensity_df = pd.read_csv("test_brake_intensity.csv")
sample_duration_df = pd.read_csv("test_brake_duration.csv")

intensity_csv = sample_intensity_df.to_csv(index=False).encode('utf-8')
duration_csv = sample_duration_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button("Download Brake Intensity Sample CSV", intensity_csv, "sample_brake_intensity.csv")
st.sidebar.download_button("Download Brake Duration Sample CSV", duration_csv, "sample_brake_duration.csv")

# Sample test data section
st.header("📊 Evaluate on Sample/Test Data with Known Labels")
if selected_model == "Brake Intensity":
    if data_option == "Upload Your Own Data":
        sample_data_file = st.file_uploader("Upload a CSV with known features and Brake Intensity labels (target column last)", type=["csv"], key="intensity_sample")
        if sample_data_file:
            test_df = pd.read_csv(sample_data_file)
        else:
            st.warning("Please upload your own test file.")
            test_df = None
    else:
        test_df = sample_intensity_df
        st.info("Using default Brake Intensity sample data.")
else:
    if data_option == "Upload Your Own Data":
        sample_data_file = st.file_uploader("Upload a CSV with known features and Brake Duration labels (target column last)", type=["csv"], key="duration_sample")
        if sample_data_file:
            test_df = pd.read_csv(sample_data_file)
        else:
            st.warning("Please upload your own test file.")
            test_df = None
    else:
        test_df = sample_duration_df
        st.info("Using default Brake Duration sample data.")

if test_df is not None:
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

    st.write("Sample Test Data Preview:")
    st.dataframe(test_df.head())

    X_test = test_df.iloc[:, :-1]
    y_true = test_df.iloc[:, -1]

    model = model_intensity if selected_model == "Brake Intensity" else model_duration

    try:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        report = classification_report(y_true, preds, output_dict=True)

        st.subheader(f"✅ {selected_model} Model Performance")
        st.write(f"**Accuracy:** {acc:.2f}")

        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())

        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.write("### 📉 Receiver Operating Characteristic (ROC) Curve")

        if proba.size > 0:
            if selected_model == "Brake Duration":
                fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve - {selected_model}')
                ax_roc.legend(loc='lower right')
                st.pyplot(fig_roc)
            else:
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                fig_mc, ax_mc = plt.subplots()
                for i in range(y_true_bin.shape[1]):
                    if i < proba.shape[1]:
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax_mc.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
                ax_mc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax_mc.set_xlabel('False Positive Rate')
                ax_mc.set_ylabel('True Positive Rate')
                ax_mc.set_title('Multi-Class ROC Curve')
                ax_mc.legend(loc='lower right')
                st.pyplot(fig_mc)

    except Exception as e:
        st.error(f"Prediction failed for {selected_model}. Ensure correct format. Error: {e}")

# User data prediction
st.header("🧾 Predict on Unseen Data (Without Labels)")
user_data_file = st.file_uploader("Upload a CSV containing only features (without target labels) to get predictions", type=["csv"], key="user")

if user_data_file:
    user_df = pd.read_csv(user_data_file)
    user_df = user_df.loc[:, ~user_df.columns.str.contains('^Unnamed')]
    st.write("Your Uploaded Data Preview:")
    st.dataframe(user_df.head())

    try:
        model = model_intensity if selected_model == "Brake Intensity" else model_duration
        user_preds = model.predict(user_df)
        user_df[f'{selected_model} Prediction'] = user_preds

        st.write("Prediction Results:")
        st.dataframe(user_df)
    except Exception as e:
        st.error(f"Prediction failed. Ensure uploaded data is in correct format. Error: {e}")

# Data schema / input instructions
st.sidebar.header("📌 Input Feature Requirements")
st.sidebar.markdown("""
Please ensure your uploaded file follows the format below:

- The file must be in CSV format
- All required feature columns must be present
- No missing values should be included

Feature Descriptions:
- `darkness`: 0 (day) or 1 (night)
- `rainy`: 0 (no rain) or 1 (rainy)
- `throttle_position`: position of accelerator pedal
- `average_speed`: mean vehicle speed
- `std_speed`: standard deviation of vehicle speed
- `relative_distance`: distance to lead vehicle
- `relative_speed`: speed difference from lead vehicle
- `relative_acc`: acceleration difference
- `time_of_max_a_to_conflict`: timing of peak acceleration before conflict
- `max_a_before_conflict`: max acceleration before conflict
- `quasi_time_head_way`: time headway approximation
- `normal_acc`: expected normal acceleration
- `normal_dec`: expected normal deceleration
- `roll_rate`, `pitch_rate`, `yaw_rate`: vehicle dynamic measurements
- Target column: `brake_duration` or `brake_intensity` in the last column
""")
