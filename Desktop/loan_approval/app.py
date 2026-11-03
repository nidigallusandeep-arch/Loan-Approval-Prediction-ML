
import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path

# ---------------------------------
# 1. Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ---------------------------------
# 2. Professional Background
# ---------------------------------
def add_background(image_path):
    if image_path.exists():
        with open(image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()

        background_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: 100% auto;
            background-position: top center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #eaf5ff;
        }}
        </style>
        """
    else:
        background_css = """
        <style>
        .stApp {
            background: linear-gradient(
                135deg, #eaf5ff, #d5eaff, #ffffff
            );
        }
        </style>
        """

    st.markdown(background_css, unsafe_allow_html=True)


add_background(BASE_DIR / "background.png")

# ---------------------------------
# 3. Custom CSS
# ---------------------------------
st.markdown("""
<style>
.stMainBlockContainer {
    padding-top: 230px;
    padding-bottom: 40px;
    max-width: 1250px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid #d8e7f7;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(20,60,100,0.08);
}

.stButton > button {
    background: linear-gradient(90deg, #0754a5, #147ac0);
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #063f80;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #dce8f5;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 4. Load Trained Model
# ---------------------------------
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "loan_approval_gradient_boosting.pkl"
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ---------------------------------
# 5. Applicant Information Form
# ---------------------------------
st.info("Enter applicant information below to predict loan approval status.")

st.header("👤 Applicant Information")

with st.container(border=True):

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
                [60, 120, 180, 240, 300, 360, 480]
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


# ---------------------------------
# 6. Prediction
# ---------------------------------
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

        classes = list(model.classes_)

        approved_index = classes.index(1)
        rejected_index = classes.index(0)

        approval_probability = probabilities[approved_index] * 100
        rejection_probability = probabilities[rejected_index] * 100

        # ---------------------------------
        # 7. Display Results
        # ---------------------------------
        st.divider()

        st.header("📊 Loan Prediction Result")

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

        st.subheader("📈 Prediction Probability")

        probability_df = pd.DataFrame({
            "Prediction": ["Approved", "Rejected"],
            "Probability (%)": [
                approval_probability,
                rejection_probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Prediction")
        )

        st.subheader("📋 Applicant Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: This is an educational machine learning "
            "demonstration. Predictions are estimates and do "
            "not guarantee actual bank approval."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ---------------------------------
# 8. Footer
# ---------------------------------
st.divider()

st.markdown("""
<center>
<p style="color:#173d70; font-weight:bold;">
🏦 AI Loan Approval Prediction
</p>
<p style="color:#52677d;">
Developed using Python, Pandas, Scikit-learn
and Gradient Boosting Classifier.
</p>
<p style="color:#52677d;">
Educational Demonstration Only.
</p>
</center>
""", unsafe_allow_html=True)



