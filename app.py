# ============================================================
# app.py  —  STREAMLIT FRONTEND FOR DIABETES PREDICTION
# ============================================================

import streamlit as st
import pandas as pd

from backend import (
    MODEL_PATH,
    download_dataset,
    evaluate_model,
    load_data,
    load_saved_model,
    plot_age_bmi_scatter,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curve,
    # plot_single_tree,
    preprocess_data,
    split_data,
    train_and_save_model,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered",
)

# ============================================================
# CUSTOM DARK THEME CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #050816, #0b1b34);
    color: white;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #081120;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #0072ff, #00c6ff);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0052cc, #0099cc);
    color: white;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

.success-box {
    background-color: rgba(46, 204, 113, 0.2);
    border: 2px solid #2ecc71;
}

.danger-box {
    background-color: rgba(231, 76, 60, 0.2);
    border: 2px solid #e74c3c;
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<h1 style='text-align:center;'>
🩺 Diabetes Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:#cccccc;'>
Enter patient health information to predict diabetes risk.
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return load_saved_model()

    model, _ = train_and_save_model()
    return model


try:
    model = get_model()
except Exception as exc:
    st.error(
        "Model is not ready yet. Add Kaggle credentials in .env and run "
        "`python backend.py` once to train and save the model."
    )
    st.exception(exc)
    st.stop()


@st.cache_resource
def get_graph_data():
    csv_path = download_dataset()
    df = load_data(csv_path)
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, 20, 42)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    return df, df_clean, results

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 📋 Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    hba1c = st.slider(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    glucose = st.slider(
        "Blood Glucose Level",
        min_value=50,
        max_value=350,
        value=120
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    smoking = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "not current", "ever", "No Info"]
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

# ============================================================
# ENCODING
# ============================================================

gender_encoded = 0 if gender == "Female" else 1

smoking_mapping = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5,
}

smoking_encoded = smoking_mapping[smoking]

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Diabetes Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender_encoded": gender_encoded,
        "smoking_encoded": smoking_encoded,
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:

        st.markdown(f"""
        <div class="prediction-box danger-box">
            🚨 High Risk of Diabetes<br><br>
            Probability: {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="prediction-box success-box">
            ✅ Low Risk of Diabetes<br><br>
            Probability: {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # METRICS
    # ========================================================

    st.markdown("## 📈 Health Metrics")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>BMI</h3>
            <h2>{bmi}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Glucose</h3>
            <h2>{glucose}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>HbA1c</h3>
            <h2>{hba1c}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Age</h3>
            <h2>{age}</h2>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# BACKEND GRAPHS
# ============================================================

st.markdown("## 📊 Model Graphs")

try:
    df, df_clean, results = get_graph_data()

    graph_tabs = st.tabs([
        "Class Distribution",
        "Correlation",
        "Confusion Matrix",
        "Feature Importance",
        "ROC Curve",
        "Age vs BMI",
        "Decision Tree",
    ])

    with graph_tabs[0]:
        st.pyplot(plot_class_distribution(df), use_container_width=True)

    with graph_tabs[1]:
        st.pyplot(plot_correlation_heatmap(df_clean), use_container_width=True)

    with graph_tabs[2]:
        st.pyplot(
            plot_confusion_matrix(results["conf_matrix"]),
            use_container_width=True,
        )

    with graph_tabs[3]:
        feature_fig, _, _ = plot_feature_importance(model)
        st.pyplot(feature_fig, use_container_width=True)

    with graph_tabs[4]:
        st.pyplot(
            plot_roc_curve(
                results["roc_fpr"],
                results["roc_tpr"],
                results["auc_score"],
            ),
            use_container_width=True,
        )

    with graph_tabs[5]:
        st.pyplot(plot_age_bmi_scatter(df_clean), use_container_width=True)

    # with graph_tabs[6]:
    #     st.pyplot(plot_single_tree(model), use_container_width=True)

except Exception as exc:
    st.warning("Graphs could not be loaded right now.")
    st.caption(str(exc))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#aaaaaa;'>

Built with ❤️ using Streamlit + Random Forest Machine Learning

</div>
""", unsafe_allow_html=True)
