# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a machine learning model where we have chosen Random Forest Classifier.

## Intended Use

## Training Data
The training data was 80% of the total dataset. We processed the data in which we one hot encoded categorical features, and employed a label binarizer for the labels.

## Evaluation Data
We evaluated the data on the remaining 20% split of the total dataset not used in training. This data was evaluated using our metrics, and also evaluated on slices of each categorical feature.

## Metrics
The metrics that we used for our model were:
* Precision
* Recall
* FBeta

Below are the metrics for both the train and test split:

| Dataset | Precision | Recall | FBeta |
|---------|-----------|--------|-------|
| Train   | 1.00       | 1.00    | 1.00   |
| Train   | 0.76     | 0.61  | 0.67 |


These metrics evaluates the performance of the model, taking into consideration the true positives and false positives.
Our model achieved:

## Ethical Considerations

## Caveats and Recommendations
