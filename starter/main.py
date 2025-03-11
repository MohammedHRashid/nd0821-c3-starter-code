from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel, Field

# Declare data obect

class Data(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str 
    education_num: int 
    marital_status: str 
    occupation: str
    relationship: str 
    race: str 
    sex: str
    capital_gain: int 
    capital_loss: int
    hours_per_week: int 
    native_country: str
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
