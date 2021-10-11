[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Monitoring Model Deterioration under DistributionShift with Explainable Uncertainty Estimation

## Abstract
Detecting distribution shift in machine learning models once they are deployed is challenging. It is even more challenging deciding when to retrain models in real-case scenarios when labeled data is beyond reach, and monitoring performance metrics is unfeasible. 
In this work, we use non-parametric bootstrapped uncertainty estimates and SHAP values to provide explainable uncertainty estimation as a technique that aims to monitor machine learning models in deployment environments and address the source of model degradation. 
We release the open-source code used to reproduce our experiments.



    
 - We use non-parametric uncertainty estimates to detect distribution shifts that cause model degradation and show that these estimates outperform classical statistics indicators in terms of minimizing false positives.
    
 - We use explainable AI techniques to identify the source of uncertainty and model deterioration for distributions and individual samples, where classical statistical indicators can only determine distribution differences between source and test datasets.


## Experiments

Our experiments have been organized into two main groups: Firstly, we assess the performance of our proposed uncertainty method for monitoring distribution drift. Secondly, we evaluate the usability of the explainable uncertainty for identifying the features that are driving model degradation in a local and global scenarios. All experiments were run on the CPU of a single MacBook Pro, and in all experiments we used the default hyperparameters in scikit-learn.



## Detecting gradual distribution shift
The notebook for this experiment is 
GradualDistShift.ipynb

## Detecting the source of uncertainty
The notebook for this experiment is xAIUncertainty.ipynb
