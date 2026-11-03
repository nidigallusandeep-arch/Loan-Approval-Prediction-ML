
import os
import joblib
import streamlit as st
import pandas as pd

# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="AI Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# 2. PROFESSIONAL UI DESIGN
# =========================================

st.markdown("""
<style>

/* Remove all background images */
.stApp {
    background: linear-gradient(
        135deg,
        #eaf2ff 0%,
        #f5f8ff 50%,
        #e6f4f1 100%
    ) !important;
    background-image: none !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Main container */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title */
.main-title {
    text-align: center;
    color: #16345c;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #52657d;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Form design */
div[data-testid="stForm"] {
    background: #ffffff !important;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #dce6f5;
    box-shadow: 0 5px 20px rgba(20,50,90,0.08);
}

/* Prediction button */
div[data-testid="stFormSubmitButton"] button {
    background-color: #1769aa !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    min-height: 48px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #104d80 !important;
    color: white !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dce6f5;
}

h1, h2, h3 {
    color: #16345c;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 3. MODEL FILE PATH
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "loan_approval_gradient_boosting.pkl"
)


# =========================================
# 4. LOAD TRAINED MODEL
# =========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Check your project folder.")
    st.stop()

try:
    model = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =========================================
# 5. APPLICATION HEADER
# =========================================

st.markdown(
    '<div class="main-title">'
    '🏦 AI Loan Approval Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Loan Eligibility Prediction Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter applicant information below to predict loan approval status."
)


# =========================================
# 6. APPLICANT INFORMATION FORM
# =========================================

st.subheader("👤 Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=500
        )

    with col2:

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0,
            step=500
        )

        loan_amount = st.number_input(
            "Loan Amount (in thousands)",
            min_value=1,
            value=150,
            step=10
        )

        loan_term = st.selectbox(
            "Loan Amount Term (months)",
            [360, 300, 240, 180, 120, 60]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# =========================================
# 7. LOAN PREDICTION
# =========================================

if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Training target encoding:
        # Rejected = 0
        # Approved = 1

        approved_index = list(model.classes_).index(1)

        approval_probability = (
            probabilities[approved_index] * 100
        )

        rejection_probability = (
            100 - approval_probability
        )

        st.divider()

        st.subheader("📊 Loan Prediction Result")

        if prediction == 1:

            st.success("🎉 Loan Approved!")

        else:

            st.error("❌ Loan Rejected!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Rejection Probability",
                f"{rejection_probability:.2f}%"
            )

        st.progress(
            float(approval_probability / 100)
        )

        st.subheader("📋 Applicant Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# =========================================
# 8. FOOTER
# =========================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#52657d;
    padding:15px;
">
    <b>AI Loan Approval Prediction</b><br>
    Developed using Python, Pandas, Scikit-learn
    and Gradient Boosting Classifier.<br>
    Educational demonstration only.
</div>
""", unsafe_allow_html=True)


