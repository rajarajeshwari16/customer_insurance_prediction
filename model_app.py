# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel, validator
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI()

# Define the input schema using Pydantic
class Input(BaseModel):
    id: int
    Gender: str
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int

    # Optional validation (e.g., to check specific ranges or values)
    @validator("Driving_License", "Previously_Insured", pre=True)
    def validate_binary(cls, value):
        if value not in [0, 1]:
            raise ValueError("Value must be 0 or 1")
        return value

    @validator("Age")
    def validate_age(cls, age):
        if age < 18 or age > 100:
            raise ValueError("Age must be between 18 and 100")
        return age

class Output(BaseModel):
    Response: int

# Load the pre-trained model
try:
    model = joblib.load("insurance_pipeline_model.pkl")
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure 'insurance_pipeline_model.pkl' exists.")

# Define the prediction endpoint
@app.post("/predict", response_model=Output)
def predict(data: Input) -> Output:
    if model is None:
        return {"error": "Model is not loaded."}

    # Create a DataFrame from the input
    X_input = pd.DataFrame([data.dict()])

    # Drop the 'id' column as it's not used for prediction
    X_input = X_input.drop(columns=["id"])

    # Make a prediction using the loaded model
    prediction = model.predict(X_input)

    # Return the output
    return Output(Response=int(prediction[0]))

# Example endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Insurance Prediction API is up and running!"}
