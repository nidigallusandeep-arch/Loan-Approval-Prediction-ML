
import os
import base64
import joblib
import streamlit as st
import pandas as pd

# --------------------------------
# 1. Page Configuration
# --------------------------------
st.set_page_config(
    page_title="AI Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------
# 2. Project File Paths
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "loan_approval_gradient_boosting.pkl"
)

BACKGROUND_PATH = os.path.join(
    BASE_DIR, "background.png"
)

# --------------------------------
# 3. Background Image
# --------------------------------
def add_bg_from_local(image_file):
    if not os.path.exists(image_file):
        st.warning("Background image not found. Using default background.")
        return

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,0.82),
                                rgba(255,255,255,0.82)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main-title {{
            color: #123456;
            text-align: center;
            font-size: 38px;
            font-weight: bold;
        }}

        .subtitle {{
            text-align: center;
            color: #333333;
            font-size: 17px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local(BACKGROUND_PATH)

# --------------------------------
# 4. Load Trained Model
# --------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# --------------------------------
# 5. Application Header
# --------------------------------
st.markdown(
    '<div class="main-title">🏦 AI Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict loan approval status using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter applicant details below to predict the loan approval status."
)

# --------------------------------
# 6. Applicant Input Form
# --------------------------------
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
        "Predict Loan Approval",
        use_container_width=True
    )

# --------------------------------
# 7. Prediction
# --------------------------------
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

        probability = model.predict_proba(input_data)[0]

        # Model target encoding:
        # Rejected = 0, Approved = 1
        if prediction == 1:
            st.success("🎉 Loan Approved!")
        else:
            st.error("❌ Loan Rejected!")

        approval_probability = probability[1] * 100

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

        st.progress(float(probability[1]))

        st.subheader("Prediction Details")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------------
# 8. Footer
# --------------------------------
st.divider()

st.caption(
    "Developed as a Machine Learning project | "
    "Gradient Boosting Classifier | Educational demonstration only"
)
