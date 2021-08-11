[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Monitoring Uncertainty - Outline

## Intro
The performance of machine learning models in production degrades over time.
In order to maintain high performance, models are retrained using previous and new input data. 
This is process is called continual learning, and it can be computationally expensive and have high demands in the software engineering system.

The performance of a model is typically judged based on the evaluation of the predictions in the deployment scenario. 
In many cases having the true label of the deployed data is not feasible. Thus is not possible to calculate evaluation metrics in order to assess the performance of the model and decide whether to retrain.

*Data is not static, it evolves*.
This fact is called distribution shift and it's the main source of model performance deterioration. 

Traditional ways of monitoring distribution shift when the real target distribution is not available are using the
Population Stability Index (PSI) or the Kolmogorov-Smirnov (KS) test. This statistical test correctly detects univariate changes
on the distribution but fails to detect when the model performance drops.

## Previous Work
Neil Lawrence - Continual Learning (NIPS2018) \
Joaquin Quiñonero - Data Drifts (NIPS2006) \
Evaluating Predictive Uncertainty Under Dataset Shift (NIPS2019)

Our work differs:
 - Tackling Dataset Shift with Uncertainty -- Not done to the best of our knowledge
 - Detecting when the model performance is downgraded. Classic statistics only detect covariate shifts.

## Experimental Methodology
Implementation and experiments can be found in a publicly available repository

### Datasets
Describe dataset, regression, continual variable

### Uncertainty Estimation
Bootstrap \
Quantile regression \
Qagging Forest? \


Brief comment why Tree methods don't work
### Pipeline
Distribution Shift \
Feature Selection \
Rebasing

### Detecting the source of the uncertainty

Using feature relevance techniques as SHAP to detect the source of the uncertainty.
## Results and Discussion
Results of uncertainty vs classical statistics

Uncertainty estimates covariate shift, furthermore, it denotes when the performance of an ML model has deteriorated.
