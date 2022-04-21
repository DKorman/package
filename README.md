# Machine learning pipeline

## Description

A machine learning pipeline is a way to codify and automate the workflow it takes to produce a machine learning model. Machine learning pipelines consist of multiple sequential steps that do everything from data extraction and preprocessing to model training and deployment.

In Clstrlobe, we have decoupled the process of Feature engineering and Model building. This private Clstrlobe library, ml_pipeline, focuses on the model building steps: preprocessing, model building, evaluation and serialization.

## Contents

```
├───ml_pipeline
│   ├───data_exploratory_analysis
│   ├───data_preprocessing
│   │   ├───data_prep
│   │   ├───encoders
│   │   ├───imputers
│   │   ├───scalers
│   ├───data_quality_checking
│   ├───feature_engineering
│   ├───model_building
│   │   ├───model_base
│   │   ├───model_strategies
│   │   ├───model_trainers
│   ├───model_eval
│   │   ├───graphs
│   │   ├───metrics
│   ├───utility
```

* `model_eval`: contains graph and metric classes used for evaluating the model
* `model_building/`: contains code for building and serializing models
  * `model_base/`: core component of model building - contains classes that apply preprocesing, training algorithm and save the trained objects
  * `model_trainers` : contains wrapper classes for commonly used ML algorithms
  * `model_strategies`: contains classes that wrap model_base, model_trainers and model_eval classes into a training schema (e.g. K fold crossvalidation)
* `data_exploratory_analysis`: colletion of tools used for performing generalized (semi)automated EDA
* `data_preprocessing`: colletion of techniques for peforming various data transformation before modelling
* `data_quality_checking`: contains tools for checking quality of input data and logging data distributions over time
* `feature_engineering`: contains certain FE steps which must be done separately for train vs. test dataset 
* `feature_engineering`: various utility functions






