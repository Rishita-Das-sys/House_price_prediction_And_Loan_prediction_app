import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Generate synthetic loan data
def generate_loan_data(n_samples=200):
    np.random.seed(42)
    age = np.random.randint(21, 60, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    loan_amount = np.random.normal(200000, 50000, n_samples)
    
    # Simple approval logic for example purposes
    approved = (income > 40000) & (credit_score > 600) & (loan_amount < 250000)
    approved = approved.astype(int)

    return pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Approved': approved
    })

# Train model
def train_loan_model():
    df = generate_loan_data()
    X = df[['Age', 'Income', 'Credit_Score', 'Loan_Amount']]
    y = df['Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model, df

# Streamlit Web App
def main():
    st.title("Bank Loan Approval Predictor")
    st.write("Enter customer details to check loan approval likelihood.")

    model, df = train_loan_model()

    age = st.slider("Applicant Age", 18, 70, 30)
    income = st.number_input("Monthly Income (INR)", min_value=10000.0, max_value=1000000.0, value=50000.0)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    loan_amount = st.number_input("Loan Amount Requested (INR)", min_value=50000.0, max_value=1000000.0, value=200000.0)

    if st.button("Predict Loan Approval"):
        input_data = np.array([[age, income, credit_score, loan_amount]])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2%})")
        else:
            st.error(f"❌ Loan Not Approved (Confidence: {prob:.2%})")

        # Visualization
        fig = px.scatter(df, x='Credit_Score', y='Income', color=df['Approved'].map({0: 'Rejected', 1: 'Approved'}),
                         title="Historical Loan Approvals by Credit Score & Income",
                         labels={'color': 'Approval Status'})
        fig.add_scatter(x=[credit_score], y=[income],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name='Current Applicant')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
