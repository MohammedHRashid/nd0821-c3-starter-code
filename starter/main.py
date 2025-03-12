from fastapi import FastAPI
from pydantic import BaseModel, Field 
import joblib
from typing import List
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference
import os

# Loading model and encoders 
base_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(base_dir, "model", "model.pkl"))
encoder = joblib.load(os.path.join(base_dir, "model", "encoder.pkl"))
lb = joblib.load(os.path.join(base_dir, "model", "lb.pkl"))


# Declare data object with correct alias

class Data(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=186824)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num", example=9)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Machine-op-inspct")
    relationship: str = Field(..., example="Unmarried")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")
    

    

# Initialise app

app = FastAPI()

# Welcome Message
@app.get("/")
def welcome_message() -> str:
    return"Welcome User to the model's API"


# Making Predictions
@app.post("/model_predictions/")
def make_predictions(data: List[Data]) -> dict:
    #df = pd.DataFrame([item.dict(by_alias=True) for item in data])
    df = pd.DataFrame([item.dict(by_alias=True) for item in data])

    # Define categorical features
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",]

    # Process input data using same encoding as training data
    input_data, _, _, _ = process_data(df, categorical_features, encoder=encoder, lb=lb, training=False)
    print(input_data)

    # Make inference predictions
    preds = inference(model, input_data)

    # Convert predictions to back to original labels
    pred_labels = lb.inverse_transform(preds)

    return {"predictions": list(pred_labels)}
