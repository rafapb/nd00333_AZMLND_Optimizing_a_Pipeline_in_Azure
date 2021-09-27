# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

![Project Overview](/img/project_overview.png)

## Summary

This dataset contains data about direct marketing campaings (phone calls) of a Portuguese banking institution.  
The dataset used can be downloaded [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

A detailed description of the data features can be found [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing#).

We seeked to predict if a client will subscribe a term deposit (target variable y being "yes" or "no"). Therefore, this is a classification problem.
The best performing model was a Voting Ensemble ran by AutoML. This model achieved an accuracy of 91.66%.

## Scikit-learn Pipeline

After setting up the workspace and [environment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments), the script "train.py" is called. The data is then downloaded from a URL and used in the TabularDatasetFactory class to create a tabular dataset. Afterwards, the data is cleaned and preprocessed by dropping null values and encoding categorical variables. 
Then, the features and target variable are separated and splitted into train and test data. 
A Logistic Regression model is fit using the training data. 
HyperDrive is used to tune the hyperparameters of the Logistic Regression model, which are Regularization Strength (C) and Maximum Number of Iteratations (max_iter).
Azure's HyperDrive was used with a Random Parameter Sampler and an Bandit Policy for early stopping.
The Logistic Regression model with parameters C = 50 and max_iter = 300 obtains the highest accuracy, 90.91%. 

To learn more about hyperparameter tuning with Azure's HyperDrive, click [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).

### Parameter sampler

Random Parameter Sampling randomly selects hyperparameters to evaluate, which makes it converge faster than the alternatives.
Grid Parameter Sampling and Bayesian Parameter Sampling are more computationally expensive since they search over all the search space.
Moreover, Random Parameter Sampling supports both discrete and continuous hyperparameters (including choice, uniform, loguniform, normal, and lognormal), although only discrete hyperparameters were used in this pipeline.

### Early stopping policy

Bandit Policy is an early termination policy based on slack factor/slack amount and evaluation interval. 
Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

## AutoML

[Azure's AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) created a number of pipelines in parallel that tried different algorithms and parameters automatically. 

The following [AutoML configuration](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py) is used:

```ruby
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=ds,
    label_column_name="y",
    n_cross_validations=2)
```

| Parameter  | Description |
| ------------- | ------------- |
| experiment_timeout_minutes  | The maximum amount of time before the experiment terminates. In this case, AutoML will stop the experiment in 30 minutes. |
| task  | The task to be performed. In this case, it is a classification problem. |
| primary_metric  | The [metric](https://docs.microsoft.com/en-us/python/api/azureml-automl-core/azureml.automl.core.shared.constants.metric?view=azure-ml-py) that AutoML optimizes for model selection. In this case, accuracy was chosen. |
| training_data  | The training data to be used within the experiment. It should contain both training features and a label column (optionally a sample weights column). |
| label_column_name  | The name of the label column. |
| n_cross_validations  | The number of [cross validations](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits) to perform. |

The best model obtained by AutoML was a Voting Ensemble, which achieved an accuracy of 91.66%.
Azure Machine Learning Studio showed that the resulting Voting Ensemble consisted of a voting of 7 different algorithms (4 XGBoost, 1 SGD, 1 LogisticRegression, and 1 LightGBM algotithms).
One of these 7 algorithms, an XGBoost algorithm, had a weight of 0.25. All the other algorithms had a weight of 0.125.  
  
The ensemble details can easily be seen in Azure's ML Studio:

![Voting Ensemble](/img/voting_ensemble.png)

Azure's ML Studio can also show other metrics such as the confusion matrix:

![AutoML Confusion Matrix](/img/automl_confusion_matrix.png)

## Pipeline comparison

The difference in performance between the Logistic Regression model and the Azure AutoML best performing model is quite small. 
This small difference could be due to the stocasthic nature of the algorithms and the hyperparameter tuning.
Both pipelines perform the same data cleaning steps. However, AutoML automatically performs its own data transformation steps, which include data preprocessing, feature engineering, and scaling techniques.
HyperDrive only tunes one model at a time whereas AutoML can tune and compare several different models without having to specify the possible hyperameters.


## Future work

Although the two models obtained an accuracy over 90%, the dataset is quite imbalanced. 
The dataset contains 3692 observations that have a target label "yes" and 29258 observations with a target label "no".
Measuring the performance with accuracy is not very appropiate since a model that predicted only the majority class would achieve an 88.80% accuracy but would be useless.
Therefore, it would be interesting to use a different metric such as the F1-score.

More information about how to choose an evaluation metric for a classification problem with an imbalanced dataset can be found [here](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/).

This imbalance in the dataset can influence many machine learning algorithms into ignoring the minority class, which in this case, is the label "yes".
Therefore, it would be interesting to perform a random oversampling of the minority class or undersampling of the majority class in order to achieve a more balanced dataset.

More on random oversampling and undersampling for imblanced classification can be found [here](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).

Last but not least, it would also be interesting to use a [Bayseian Parameter Sampler](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.bayesianparametersampling?view=azure-ml-py) with the Logistic Regression model.
Although it would take more time to train, it would be interesting to see if a better performance could be yield using this parameter sampler.


## Proof of cluster clean up

Right after executing the last cell of code:

```ruby
compute_target.delete()
```

The compute cluster gets deleted:

![Cluster Cleanup](/img/cluster_cleanup.png)


