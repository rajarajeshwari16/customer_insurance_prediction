# pip install streamlit
import streamlit as st
import pandas as pd
import joblib

# Set the Streamlit app title
st.title("Insurance Prediction App")
st.write("Predict if a customer is interested in vehicle insurance.")

# Load the dataset to populate input options
df = pd.read_csv('train.csv')

# Create input fields for the user to provide data
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Age = st.number_input("Age", step=1, value=df['Age'].min())
Driving_License = st.selectbox("Driving License (0 = No, 1 = Yes)", [0, 1])
Region_Code = st.selectbox("Region Code", pd.unique(df['Region_Code']))
Previously_Insured = st.selectbox("Previously Insured (0 = No, 1 = Yes)", [0, 1])
Vehicle_Age = st.selectbox("Vehicle Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle Damage (Yes/No)", pd.unique(df['Vehicle_Damage']))
Annual_Premium = st.number_input("Annual Premium", step=1.0, value=df['Annual_Premium'].mean())
Policy_Sales_Channel = st.selectbox("Policy Sales Channel", pd.unique(df['Policy_Sales_Channel']))
Vintage = st.number_input("Vintage", step=1, value=int(df['Vintage'].mean()))


# Convert inputs into a dictionary for model prediction
inputs = {
    "Gender": Gender,
    "Age": Age,
    "Driving_License": Driving_License,
    "Region_Code": Region_Code,
    "Previously_Insured": Previously_Insured,
    "Vehicle_Age": Vehicle_Age,
    "Vehicle_Damage": Vehicle_Damage,
    "Annual_Premium": Annual_Premium,
    "Policy_Sales_Channel": Policy_Sales_Channel,
    "Vintage": Vintage
}

# Display input data for user confirmation
st.write("Inputs Provided:", inputs)

# Prediction button
if st.button("Predict"):
    # Load the saved model
    model = joblib.load('insurance_pipeline_model.pkl')

    # Convert input dictionary into a DataFrame
    X_input = pd.DataFrame(inputs, index=[0])

    # Generate predictions
    prediction = model.predict(X_input)

    # Display the prediction result
    result = "Customer is interested in vehicle insurance." if prediction[0] == 1 else "Customer is not interested in vehicle insurance."
    st.write("Prediction:", result)
