
import os
import base64
import joblib
import streamlit as st
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. BACKGROUND IMAGE
# ==========================================

BACKGROUND_PATH = os.path.join(
    BASE_DIR,
    "background.png"
)

if os.path.exists(BACKGROUND_PATH):

    with open(BACKGROUND_PATH, "rb") as image:

        encoded_image = base64.b64encode(
            image.read()
        ).decode()

    st.markdown(f"""
    <style>
    .hero-banner {{
        width: 100%;
        height: 270px;
        background-image: url(
            "data:image/png;base64,{encoded_image}"
        );
        background-size: cover;
        background-position: center 15%;
        background-repeat: no-repeat;
        border-radius: 15px;
        margin-bottom: 25px;
    }}
    </style>

    <div class="hero-banner"></div>

    """, unsafe_allow_html=True)

else:

    st.warning("Background image not found!")

# ==========================================
# 3. PROFESSIONAL CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eaf2ff 0%,
        #f5f8ff 50%,
        #e6f4f1 100%
    );
}

.main .block-container {
    max-width: 1200px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* Applicant Form */

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.98);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #dce6f5;
    box-shadow: 0 5px 20px rgba(20,50,90,0.08);
}

/* Submit Button */

div[data-testid="stFormSubmitButton"] button {
    background-color: #1769aa;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    min-height: 45px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #104d80;
    color: white;
}

/* Metric Cards */

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dce6f5;
    box-shadow: 0 3px 12px rgba(20,50,90,0.05);
}

/* Headings */

h1, h2, h3 {
    color: #16345c;
}

/* Footer */

.footer {
    text-align: center;
    color: #52657d;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MODEL FILE PATH
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "loan_approval_gradient_boosting.pkl"
)

# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found! Please check the project folder."
    )

    st.stop()


try:

    model = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")

    st.stop()

# ==========================================
# 6. APPLICANT INFORMATION
# ==========================================

st.info(
    "Enter applicant information below to predict loan approval status."
)

st.subheader("👤 Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    # --------------------------------------
    # LEFT COLUMN
    # --------------------------------------

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

        # Direct number typing

        applicant_income = st.text_input(
            "Applicant Income",
            value="5000",
            placeholder="Enter applicant income"
        )

    # --------------------------------------
    # RIGHT COLUMN
    # --------------------------------------

    with col2:

        # Direct number typing

        coapplicant_income = st.text_input(
            "Coapplicant Income",
            value="0",
            placeholder="Enter coapplicant income"
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

# ==========================================
# 7. LOAN PREDICTION
# ==========================================

if submitted:

    # Validate income

    if not applicant_income.isdigit():

        st.error(
            "Please enter a valid Applicant Income."
        )

        st.stop()

    if not coapplicant_income.isdigit():

        st.error(
            "Please enter a valid Coapplicant Income."
        )

        st.stop()

    # Convert income into integer

    applicant_income = int(applicant_income)

    coapplicant_income = int(coapplicant_income)

    # Prepare input data

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

        # Model prediction

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Target encoding:
        # Rejected = 0
        # Approved = 1

        classes = list(model.classes_)

        approved_index = classes.index(1)

        rejected_index = classes.index(0)

        approval_probability = (
            probabilities[approved_index] * 100
        )

        rejection_probability = (
            probabilities[rejected_index] * 100
        )

        # ==================================
        # 8. PREDICTION RESULT
        # ==================================

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

        # Approval progress

        st.write("Approval Probability")

        st.progress(
            float(probabilities[approved_index])
        )

        # ==================================
        # 9. APPLICANT DETAILS
        # ==================================

        st.subheader("📋 Applicant Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: This is an educational machine learning "
            "demonstration. Predictions are estimates and "
            "do not guarantee actual bank approval."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">

<b>🏦 AI Loan Approval Prediction</b><br>

Developed using Python, Pandas, Scikit-learn
and Gradient Boosting Classifier.<br>

Educational Demonstration Only.

</div>
""", unsafe_allow_html=True)



