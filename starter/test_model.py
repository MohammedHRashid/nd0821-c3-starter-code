import pytest
import pandas as pd
from starter.ml.data import process_data
import model
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def data():
    data = pd.read_csv("starter/data/census.csv")
    return data

@pytest.fixture
def categorical_features():
    categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    return categorical_features


@pytest.fixture
def processed_data(data, categorical_features):

    # Train test split
    train, test = train_test_split(data, test_size=0.20)


    # Process train data    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label="salary", training=True
        )

    # Process test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
        )
    return X_train, y_train, X_test, y_test

@pytest.fixture
def model():
    return joblib.load('starter/model/model.pkl')

# Testing functioms

def test_function_train_model(processed_data):
    '''
    This tests the function train_model correctly trains data to be random forest classifier
    '''
    X_train, y_train , _, _ = processed_data
    
    model = train_model(X_train,y_train)
    
    assert isinstance(model, RandomForestClassifier), "Model did not train as RandomForestClassifier"
    

def test_inference(processed_data, model):
    X_train, y_train , _, _ = processed_data
    preds = inference(model, X_train)
    
    assert preds.shape[0]>0,"Predictions have no rows"
    assert preds.shape[1]>0,"Predictions have no columns"


def test_function_compute_metrics(processed_data,model):
    '''
    This tests the function compute_metrics and assesses if metrics have been computed correctly
    '''
    X_train, y_train , _, _ = processed_data
    
    preds = inference(model,X_train)
    
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    
    assert 0<=precision<=1 , "Precision was not computed correctly"
    assert 0<=recall<=1 , "Recall was not computed correctly"
    assert 0<=fbeta<=1, "FBeta was not computed correctly"
    
    
    
    
