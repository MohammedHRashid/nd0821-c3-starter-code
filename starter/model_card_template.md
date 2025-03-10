# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a machine learning model where we have chosen Random Forest Classifier to classify salary thresholds for people in the dataset.

## Intended Use
This model was designed as part of the project for Udacity Machine Learning DevOps Engineer nanodegree.

## Training Data
The training data was 80% of the total dataset. We processed the data in which we one hot encoded categorical features, and employed a label binarizer for the labels.

## Evaluation Data
We evaluated the data on the remaining 20% split of the total dataset not used in training. This data was evaluated using our metrics, and also evaluated on slices of each categorical feature.

## Metrics
The metrics that we used for our model were:
* Precision
* Recall
* FBeta


These metrics evaluates the performance of the model, taking into consideration the true positives and false positives.
Our model achieved:

| Dataset | Precision | Recall | FBeta |
|---------|-----------|--------|-------|
| Train   | 1.00       | 1.00    | 1.00   |
| Train   | 0.76     | 0.61  | 0.67 |


## Ethical Considerations
* We have iterated the performance over categorical slices to identify whether there are any potential biases in the model. These slice performances can be found in ```slices_output.txt```.

## Caveats and Recommendations

* It appears that the data does have some null values present. To ensure the model performs better these can be handled by either dropping them or imputing values appropriately.

* We have not employed any parameter tuning and other rigorous methods to achieve a high performing model. We recommend exploring these.

* The only model we experimented with was a Random Forest Classifier

