[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Monitoring Model Deterioration with Explainable Uncertainty Estimation via Non-parametric Bootstrap
## Abstract
Monitoring machine learning models once they are deployed is challenging. It is even more challenging deciding when to retrain models in real-case scenarios when labeled data is beyond reach, and monitoring performance metrics becomes unfeasible. 
In this work, we use non-parametric bootstrapped uncertainty estimates and SHAP values to provide explainable uncertainty estimation as a technique that aims to monitor machine learning in deployment environments, and determine the source of model degradation when  target labels are not available. Classical methods are purely aimed at detecting distribution shift, which can lead to false positives in the sense that the model has not deteriorated despite a shift in the data distribution.
We show that our uncertainty estimation method more accurately measures model deterioration than the current state-of-the-art.
To estimate model uncertainty, we improve the work of  to build prediction intervals, which we show have more accurate coverage in high-variance scenarios.
Finally, we use explainable AI techniques to gain understanding on the drivers of model deterioration.
We release an open source Python package, \texttt{doubt}, which implements our proposed methods, along the code used to reproduce our experiments.


Our contributions are the following:

- We develop a novel method that produces prediction intervals using bootstrapping with theoretical guarantees, which achieves better coverage than previous methods on eight real-life regression datasets from the UCI repository \cite{uci_data}.
    
- We use non-parametric uncertainty estimates to monitor the performance of ML models in the absence of the true label in deployment scenarios and provide evidence that our non-parametric uncertainty estimation method outperforms classical statistics indicators in terms of detecting model deterioration.
    
- We use explainable AI techniques to identify the source of uncertainty and model deterioration for distributions and individual samples, where classical statistical indicators can only determine distribution differences between training and test datasets. 

-  We release an open source Python package, \texttt{doubt}, which is compatible with all \texttt{scikit-learn} models and enables prediction interval estimation using our method.


## Experiments

Our experiments have been
organized into three main groups: Firstly, we compare our non-parametric bootstrapped estimation method with the previous state of the art, NASA and MAPIE. Secondly, we assess the performance of our proposed uncertainty method for monitoring the performance of a machine learning model. And then, we evaluate the usability of the explainable uncertainty for identifying the features that are driving model degradation in local and global scenarios. 


## Uncertainty method comparison
The notebook for this experiment is 
uncertaintyExperiments.ipynb

## Evaluating model deterioration
The notebook for this experiment is 
SortedDistShift.ipynb

## Detecting the source of uncertainty
The notebook for this experiment is xAIUncertainty.ipynb

## Experiments on synthetic data

Monitoring `syntheticMonitoring.py`
Explainable AI `syntheticxAI.py`

```python
from sklearn.linear_model import LinearRegression
from doubt import Boot
import numpy as np

x1 = np.random.normal(1, 0.1, size=10000)
x2 = np.random.normal(1, 0.1, size=10000)
x3 = np.random.normal(1, 0.1, size=10000)
X = np.array([x1, x2, x3]).T
X_ood = np.array([x1 + 5, x2, x3]).T
y = x1 ** 2 + x2 + np.random.normal(0, 0.01, 10000)
clf = Boot(LinearRegression())
clf = clf.fit(X, y)

preds, intervals = clf.predict(X_ood, uncertainty=0.05)
unc = intervals[:, 1] - intervals[:, 0]

m = LinearRegression().fit(X_ood, unc)
np.round(m.coef_, decimals=2)
#[ 0.01,  0.  , -0.  ]
```
