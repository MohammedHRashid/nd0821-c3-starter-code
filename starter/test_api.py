from fastapi.testclient import TestClient
import pytest
import json
from typing import List

# Importing our API app
from main import app

# Initialising testing client
client = TestClient(app)

def test_welcome_message():
    r = client.get("/")
    
    # Testing if status code is a success
    assert r.status_code == 200
    
    # Testing if response message is as expected
    assert r.text == '"Welcome User to the model\'s API"'
    
# Testing first prediction
def test_prediction_1():
    
    row = { 
        "age": 43,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 292175,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    
    r = client.post("/model_predictions/", json=row)
    
    # Testing if status code is a success
    assert r.status_code == 200
    
    # We expect this to have prediction >50K
    expected_prediction = [">50K"]  
    assert r.json()==  {"predictions": expected_prediction}
    
# Testing second prediction
def test_prediction_2():
    
    row = { 
        "age": 39,
        "workclass": "Private",
        "fnlgt": 367260,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }
    
    r = client.post("/model_predictions/", json = [row])
    
    # Testing if status code is a success
    assert r.status_code == 200
    
    # We expect this to have prediction <=50K
    expected_prediction = ["<=50K"]  
    assert r.json()== {"predictions": expected_prediction}