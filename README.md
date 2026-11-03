# 🏦 Loan Approval Prediction Using Gradient Boosting Classifier

## 🚀 Live Demo

**Try the live application here:**

👉 [Click Here to Open Loan Approval Prediction App](https://loan-approval-prediction-mlgitaabc3243a135fbmain-mainnidagallu.streamlit.app/)

Explore the application by entering applicant details and predicting whether a loan will be approved or rejected.

---

## 📌 Project Overview

The **Loan Approval Prediction** project is a Machine Learning classification project that predicts whether a loan application will be Approved or Rejected based on applicant information.

The model uses applicant details such as income, education, marital status, credit history, loan amount, and property area to make predictions.

The trained model is deployed using Streamlit, allowing users to interact with the application and get real-time predictions.

## 🎯 Project Objectives

- Predict loan approval status using Machine Learning.
- Analyze applicant financial and personal information.
- Perform data cleaning and preprocessing.
- Apply Gradient Boosting Classifier.
- Evaluate model performance.
- Deploy the trained model using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Gradient Boosting | Classification algorithm |
| Streamlit | Web application |
| Joblib | Model serialization |
| Git & GitHub | Version control |

## 📂 Dataset Information

- Dataset size: 20,000 records
- Problem type: Binary Classification
- Target variable: `Loan_Status`
- Target classes: `Approved` and `Rejected`

### Input Features

1. Gender
2. Married
3. Dependents
4. Education
5. Self_Employed
6. ApplicantIncome
7. CoapplicantIncome
8. LoanAmount
9. Loan_Amount_Term
10. Credit_History
11. Property_Area

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Handling Missing Values
6. Categorical Feature Encoding
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Model Saving using Joblib
11. Streamlit Deployment

## 🤖 Machine Learning Algorithm

**Gradient Boosting Classifier**

Gradient Boosting is an ensemble Machine Learning algorithm that combines multiple weak learners, usually decision trees, to build a stronger predictive model.

It improves predictions by learning from the errors made by previous trees.

## 📊 Application Features

- User-friendly interface.
- Applicant information input form.
- Income and loan amount inputs.
- Credit history selection.
- Property area selection.
- Loan approval prediction.
- Approval and rejection probability.
- Real-time prediction results.

## 📈 Sample Prediction

Example applicant information:

| Feature | Value |
|---|---|
| Gender | Male |
| Married | Yes |
| Dependents | 0 |
| Education | Graduate |
| Self Employed | Yes |
| Applicant Income | 39,500 |
| Coapplicant Income | 2,500 |
| Loan Amount | 250 |
| Loan Term | 360 Months |
| Credit History | 1 |
| Property Area | Urban |

**Example application result:**

- Prediction: Loan Approved
- Approval Probability: 88.06%
- Rejection Probability: 11.94%

*Note: This is an example output from the application, not a guarantee of actual loan approval.*

## 📁 Project Structure

```text
Loan-Approval-Prediction-ML/
│
├── Desktop/
│   └── loan_approval/
│       ├── app.py
│       ├── requirements.txt
│       ├── loan_approval_gradient_boosting.pkl
│       ├── loan_approval_20000.csv
│       ├── background.png
│       └── Day_10_Loan_Approval_Prediction.ipynb
│
└── README.md
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nidigallusandeep-arch/Loan-Approval-Prediction-ML.git
```

### 2. Navigate to the project folder

```bash
cd Loan-Approval-Prediction-ML/Desktop/loan_approval
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

🔗 **Live Demo:** https://loan-approval-prediction-mlgitaabc3243a135fbmain-mainnidagallu.streamlit.app/

## 💡 Key Learnings

- Understanding classification problems.
- Data preprocessing and feature encoding.
- Handling missing values.
- Training Gradient Boosting models.
- Evaluating classification performance.
- Saving and loading trained ML models.
- Building interactive applications using Streamlit.
- Deploying Machine Learning applications.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes only. It is not an official banking or financial decision-making system. Predictions should not be used as the sole basis for real loan decisions.

## 👨‍💻 Author

**Nidigallu Sandeep**

GitHub: [nidigallusandeep-arch](https://github.com/nidigallusandeep-arch)

---

⭐ If you find this project useful, please consider giving the repository a star!

