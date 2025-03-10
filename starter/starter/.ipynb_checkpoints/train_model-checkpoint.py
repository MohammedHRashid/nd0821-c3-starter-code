# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
import os
from pathlib import Path
import pandas as pd
from ml.model import train_model
import joblib

# Add the necessary imports for the starter code.

# Get the parent directory of the current script
parent_dir = Path(__file__).resolve().parent.parent

# Add code to load in the data.
data_path = os.path.join(parent_dir, 'data/census.csv')
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
model_dir =  os.path.join(parent_dir,'model')

model = train_model(X_train, y_train)
joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

# Save OneHotEncoder
joblib.dump(encoder, os.path.join(model_dir, 'encoder.pkl'))

# Save LabelBinarizer
joblib.dump(lb, os.path.join(model_dir, 'lb.pkl'))