from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel, Field 
import joblib
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

# Loading model

model = joblib.load("model/model.pkl")
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load("model/lb.pkl")


# Declare data object

class Data(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str 
    education_num: int = Field(..., alias="education-num")
    marital_status: str 
    occupation: str
    relationship: str 
    race: str 
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int  = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")
    salary: str
    
model_config = {
        "json_schema_extra": {
            "example": { 
                "age": 32,
                "workclass": "Private",
                "fnlgt": 186824,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Never-married",
                "occupation": "Machine-op-inspct",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
                "salary": "<=50K",
            }
        }
    }
    

# Initialise app

app = FastAPI()

# Welcome Message
@app.get("/")
def welcome_message() -> dict:
    return {"message": "Welcome User to the model's API"}

# Fixing data

@app.post("/fixed_data/")
def prepare_data_for_model(data: list[Data]):  
    # Relabelling columns with hyphens
    df = pd.DataFrame([item.dict(by_alias=True) for item in data])
    return df.to_dict(orient="records")

# Making predictions

@app.post('/model_predictions/')
def make_predictions()
