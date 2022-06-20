# %%
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBClassifier
from scipy.stats import kstest
import shap
import numpy as np
import sys
from mapie.regression import MapieRegressor



sys.path.append("../")
import random

random.seed(0)

import matplotlib.pyplot as plt
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["HI"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)

# OOD
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["NY"], download=True)
mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)
mi_features = mi_features.head(2_000)
mi_labels = mi_labels[:2_000]
# %%
# Modeling
X_tr, X_te, y_tr, y_te =  train_test_split(ca_features, ca_labels, test_size=0.33, random_state=0)
regressor = LinearRegression()
alpha = [0.05, 0.95]
mapie = MapieRegressor(regressor)
X_te = mi_features
mapie.fit(X_tr, y_tr)
preds, intervals = mapie.predict(X_te, alpha=alpha)


# %%
mean_width = np.mean(intervals[:, 1] - intervals[:, 0],axis=1)
mean_width = mean_width.reshape(-1, 1)

# %%
sc = StandardScaler()
aux = X_te.copy()
aux['preds'] = preds
aux['unc'] = sc.fit_transform(mean_width)
aux['unc'] = aux['unc'].rolling(window=100).mean().fillna(0)
aux['error'] = sc.fit_transform(np.abs(preds-mi_labels).reshape(-1, 1))
#aux['error'] = aux['error'].rolling(window=10).mean().fillna(0)
aux = aux.sort_values(by='error')
aux = aux.reset_index()

# %%
plt.figure()
plt.plot(aux.unc,label='uncertainty')
plt.plot(aux.error,label='error')
plt.legend()