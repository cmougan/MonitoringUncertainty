# %%
# General imports
import numpy as np
import matplotlib.pyplot as plt
from doubt import Boot
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

plt.style.use("seaborn-whitegrid")
from scipy.stats import ks_2samp, kstest
import seaborn as sns
import pandas as pd
import random
from collections import defaultdict

# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression

# Specific packages
import shap

# Seeding
np.random.seed(0)
random.seed(0)
# %%

## Create variables
### Normal
samples = 10_000
x1 = np.random.normal(1, 5, size=samples)
x2 = np.random.normal(1, 5, size=samples)
x3 = np.random.normal(1, 5, size=samples)
mean = (1, 1, 1)
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# x = np.random.multivariate_normal(mean, cov, size=samples)
# df = pd.DataFrame(x)


df = pd.DataFrame(data=[x1, x2, x3]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
df["target"] = df["Var1"] + 0.1 * df["Var2"] + np.random.normal(0, 0.1, samples)
# %%
## Fit our ML model
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])

X_ood = X_te.copy()
for col in X_ood.columns:
    X_ood[col] = X_ood[col] + 100
# %%
l = Lasso(alpha=0.001).fit(X_tr, y_tr)
model = Boot(LinearRegression())
model.fit(X_tr, y_tr.target.values, n_boots=20)

# Predictions
preds, intervals = model.predict(X_ood, uncertainty=0.05, n_boots=20)
uncertainty = intervals[:, 1] - intervals[:, 0]
# %%

g = LinearRegression()
g.fit(X_ood, uncertainty)

## Explanation
explainer = shap.LinearExplainer(
    g, X_ood, feature_dependence="interventional"
)  # "correlation_dependent"
shap_values = explainer(X_ood)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
)
# %%

np.abs(exp).mean()
# %%
exp.head(1)
# %%
g.coef_
# %%
