from fastapi import FastAPI
from pydantic import BaseModel, Field 
import joblib
from typing import List
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

# Loading model and encoders 
model = joblib.load("model/model.pkl")
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load("starter/model/lb.pkl")


# Declare data object with correct alias

class Data(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str 
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str 
    race: str 
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int  = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")
    
    # Example data sample
    model_config = {
        "json_schema_extra": {
            "example": { 
                "age": 32,
                "workclass": "Private",
                "fnlgt": 186824,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Never-married",
                "occupation": "Machine-op-inspct",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
    }
    

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
    df = pd.DataFrame([item.dict(by_alias=True, exclude={"model_config"}) for item in data])

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
