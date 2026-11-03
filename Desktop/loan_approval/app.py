import base64
import streamlit as st

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background: rgba(255, 255, 255, 0.90);
            padding: 2rem;
            border-radius: 20px;
            margin-top: 25px;
        }}

        h1 {{
            color: #102A66;
            text-align: center;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.png")

st.title("🏦 Loan Approval Prediction")

st.write(
    "Predict whether a loan application will be approved or rejected."
)