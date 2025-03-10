from ml.data import process_data
from ml.model import compute_model_metrics, inference


def slice_performance_metrics(data, model, encoder, lb, cat_features, save_path):
    '''
    This function produces performance metrics of a model to the dataset using feature slices of the data.
    The arguments of the model include the data, the model and encoder and labelbinarizer used to train the model,
    a list of the categorical features, and save_path which is the path you wish to save the metrics to.
    '''
    
    # Process the test data with the process_data function.
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    with open(save_path, 'w') as f:
        
        # Iterating over each categorical feature
        for feature in cat_features:
            f.write(f"\nFeature: {feature}: \n")
            
            # List containing unique values existing in categorical feature
            slices = list(data[feature].unique())
            
            for feature_slice in slices:
                
                # Data sliced to only contain feature_slice
                data_slice = data[data[feature] == feature_slice]
            
                X_slice, y_slice ,_,_ = process_data(
                    data_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
                    )
            
                # Predictions made on feature slice
                preds_slice = inference(model, X_slice)
            
            
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
                f.write(f"Slice: {feature_slice}, Precision: {precision:.2f}, Recall: {recall:.2f}, FBeta: {fbeta:.2f} \n")
