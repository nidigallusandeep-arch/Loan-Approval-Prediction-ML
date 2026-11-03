
import os
import joblib
import streamlit as st
import pandas as pd

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==================================================
# 2. CUSTOM CSS - LIGHT BLUE GRADIENT THEME
# ==================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #F8FAFC 0%,
        #E0F2FE 50%,
        #DBEAFE 100%
    );
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Headings */
h1, h2, h3 {
    color: #12356B !important;
    font-weight: 700;
}

/* General Text */
.stApp p,
.stApp label {
    color: #243B53;
}

/* Form Container */
div[data-testid="stForm"] {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #CBDFF5;
    box-shadow: 0 5px 20px rgba(30, 64, 175, 0.08);
}

/* Form Labels */
div[data-testid="stForm"] label,
div[data-testid="stForm"] p {
    color: #243B53 !important;
}

/* Input Fields */
div[data-testid="stForm"] input {
    background-color: #F1F5F9;
    color: #1E293B;
    border-radius: 7px;
}

/* Selectbox */
div[data-testid="stForm"] div[data-baseweb="select"] {
    background-color: #F1F5F9;
    border-radius: 7px;
}

/* Predict Button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(
        90deg,
        #0759C7,
        #2563EB
    );
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

/* Button Hover */
div[data-testid="stFormSubmitButton"] button:hover {
    background: #12356B;
    color: white;
}

/* Metrics Cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #D6E5F5;
    box-shadow: 0 3px 10px rgba(30, 64, 175, 0.05);
}

/* Metric Text */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: #12356B !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 10px;
}

/* Progress Bar */
div[data-testid="stProgressBar"] > div {
    background-color: #DBEAFE;
}

/* Footer */
.footer {
    text-align: center;
    color: #12356B;
    font-size: 13px;
    padding: 15px;
}

/* Info Message */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. MODEL FILE PATH
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "loan_approval_gradient_boosting.pkl"
)


# ==================================================
# 4. LOAD MACHINE LEARNING MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==================================================
# 5. HEADER
# ==================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🏦 AI Loan Approval Prediction
    </h1>

    <p style="
        text-align:center;
        color:#315B8A;
        font-size:16px;
    ">
        Smart Loan Eligibility Prediction Using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# 6. APPLICANT INFORMATION FORM
# ==================================================

st.subheader("📋 Enter Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    # ---------------- LEFT COLUMN ----------------

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

        applicant_income = st.text_input(
            "Applicant Income",
            value="5000",
            placeholder="Enter applicant income"
        )

    # ---------------- RIGHT COLUMN ----------------

    with col2:

        coapplicant_income = st.text_input(
            "Coapplicant Income",
            value="0",
            placeholder="Enter coapplicant income"
        )

        loan_amount = st.number_input(
            "Loan Amount (in thousands)",
            min_value=1.0,
            value=150.0,
            step=5.0
        )

        loan_term = st.selectbox(
            "Loan Amount Term (Months)",
            [60, 120, 180, 240, 300, 360, 480],
            index=5
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x:
            "Good (1.0)" if x == 1.0 else "Bad (0.0)"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    st.write("")

    submit = st.form_submit_button(
        "🚀 Predict Loan Approval",
        use_container_width=True
    )


# ==================================================
# 7. PREDICTION
# ==================================================

if submit:

    # Validate Income

    if not applicant_income.isdigit():

        st.error("Please enter a valid Applicant Income.")
        st.stop()

    if not coapplicant_income.isdigit():

        st.error("Please enter a valid Coapplicant Income.")
        st.stop()

    applicant_income = int(applicant_income)
    coapplicant_income = int(coapplicant_income)

    if applicant_income <= 0:

        st.error("Applicant Income must be greater than zero.")
        st.stop()

    # Create Input DataFrame

    input_data = pd.DataFrame({

        "Gender": [gender],

        "Married": [married],

        "Dependents": [dependents],

        "Education": [education],

        "Self_Employed": [self_employed],

        "ApplicantIncome": [applicant_income],

        "CoapplicantIncome": [coapplicant_income],

        "LoanAmount": [loan_amount],

        "Loan_Amount_Term": [loan_term],

        "Credit_History": [credit_history],

        "Property_Area": [property_area]

    })

    try:

        # Model Prediction

        prediction = model.predict(input_data)[0]

        # Prediction Probabilities

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        approved_index = classes.index(1)
        rejected_index = classes.index(0)

        approval_probability = float(
            probabilities[approved_index]
        )

        rejection_probability = float(
            probabilities[rejected_index]
        )


        # ==========================================
        # 8. PREDICTION RESULT
        # ==========================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        # Balloons only after successful prediction

        st.balloons()

        if prediction == 1:

            st.success(
                "🎉 Loan Application: APPROVED"
            )

        else:

            st.error(
                "❌ Loan Application: REJECTED"
            )


        # ==========================================
        # 9. PROBABILITY METRICS
        # ==========================================

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                label="Approval Probability",
                value=f"{approval_probability * 100:.2f}%"
            )

        with metric2:

            st.metric(
                label="Rejection Probability",
                value=f"{rejection_probability * 100:.2f}%"
            )


        # ==========================================
        # 10. APPROVAL PROGRESS BAR
        # ==========================================

        st.subheader("📊 Approval Probability")

        st.progress(approval_probability)


        # ==========================================
        # 11. APPLICANT DETAILS
        # ==========================================

        st.subheader("👤 Applicant Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


        # ==========================================
        # 12. DISCLAIMER
        # ==========================================

        st.info(
            "This prediction is generated by a Machine "
            "Learning model using a synthetic dataset. "
            "It is an educational demonstration and not "
            "an actual bank decision."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        Developed using Python | Pandas | Scikit-learn |
        Gradient Boosting Classifier | Streamlit

        <br><br>

        Educational Demonstration Only

    </div>
    """,
    unsafe_allow_html=True
)



