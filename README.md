# Machine learning pipeline

## Description

A machine learning pipeline is a way to codify and automate the workflow it takes to produce a machine learning model. Machine learning pipelines consist of multiple sequential steps that do everything from data extraction and preprocessing to model training and deployment.

In Clstrlobe, we have decoupled the process of Feature engineering and Model building. This private Clstrlobe library, ml_pipeline, focuses on the model building steps: preprocessing, model building, evaluation and serialization.

## Contents

```
├── .github
│   └── workflows
│       ├── production.yaml
│       └── staging.yaml


'''
