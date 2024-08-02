import streamlit as st
import joblib
import numpy as np
from streamlit_option_menu import option_menu

# Load the trained model
model = joblib.load('dt_model_compressed_high.pkl')

# Function to get user inputs
def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        Annual_Income = st.number_input('Annual Income', min_value=0.0, step=0.1)
        Num_of_Loan = st.number_input('Number of Loans', min_value=0, step=1)
        Num_of_Delayed_Payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
        Num_Credit_Inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)
        Credit_Utilization_Ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, step=0.1)
        Total_EMI_per_month = st.number_input('Total EMI per month', min_value=0.0, step=0.1)
       
    with col2:
        Interest_Rate = st.number_input('Interest Rate', min_value=0.0, step=0.1)
        Delay_from_due_date = st.number_input('Delay from due date', min_value=0.0, step=0.1)
        Changed_Credit_Limit = st.number_input('Changed Credit Limit', min_value=0.0, step=0.1)
        Outstanding_Debt = st.number_input('Outstanding Debt', min_value=0.0, step=0.1)
        Credit_History_Age = st.number_input('Credit History Age', min_value=0.0, step=0.1)
        Monthly_Balance = st.number_input('Monthly Balance', min_value=0.0, step=0.1)
                
    # Create an input array for the model
    input_data = np.array([Annual_Income, Interest_Rate, Num_of_Loan, Delay_from_due_date, 
                           Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Outstanding_Debt,
                           Credit_Utilization_Ratio, Credit_History_Age, Total_EMI_per_month, Monthly_Balance]).reshape(1, -1)
    
    return input_data

# Mapping from numeric labels to descriptive labels and colors
label_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
color_mapping = {'Poor': 'red', 'Standard': 'green', 'Good': 'green'}

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Predict Credit Score", "About"],
        icons=["house", "calculator", "info-circle"],
        menu_icon="cast",
        default_index=1,
    )

# Home page
if selected == "Home":
    st.title("Welcome to the Credit Score Classification")
    st.write("This App focuses on credit classification. The main objective is to develop a predictive model to categorize credit applicants into different risk levels. By analyzing various features and historical data of applicants, the model aims to assist financial institutions in evaluating credit applications more effectively. The classification goal is to categorize credit scores into three main groups: Standard, Poor, and Good. The project aims to improve the efficiency and accuracy of the credit approval process, reduce default risks, and optimize the overall credit management strategy.")

# Predict Credit Score page
elif selected == "Predict Credit Score":
    st.title("Credit Score Prediction")

    # Get user input
    user_input = get_user_input()

    # Center the Predict button using CSS
    st.markdown("""
        <style>
        .stButton>button {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Button to make predictions
    if st.button("Predict Credit Score"):
        # Make prediction
        numeric_prediction = model.predict(user_input)[0]
        
        # Map numeric prediction to descriptive label
        descriptive_prediction = label_mapping[numeric_prediction]
        prediction_color = color_mapping[descriptive_prediction]

        # Display the result with color
        st.markdown(f"<h2 style='color: {prediction_color};'>The predicted credit score category is: {descriptive_prediction}</h2>", unsafe_allow_html=True)

# About page
elif selected == "About":
    st.title("About")
    st.write("This app predicts credit scores category based on user inputs. Created for the ICTAK internship project.")