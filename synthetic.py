# %%
# General imports
import numpy as np
import matplotlib.pyplot as plt
from doubt import Boot

plt.style.use("seaborn-whitegrid")
import pandas as pd
import random
import seaborn as sns

# Scikit Learn
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.dummy import DummyRegressor

# Specific packages
import shap

# Seeding
np.random.seed(0)
random.seed(0)
# %%

## Create variables
### Normal
samples = 10_000
x1 = np.random.normal(1, 0.1, size=samples)
x2 = np.random.normal(1, 0.1, size=samples)
x3 = np.random.normal(1, 0.1, size=samples)

# Convert to dataframe
df = pd.DataFrame(data=[x1, x2, x3]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
df["target"] = df["Var1"] ** 2 + df["Var2"] + np.random.normal(0, 0.01, samples)
# %%
## Fit our ML model
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])

X_ood = X_te.copy()
for col in X_ood.columns:
    X_ood[col] = X_ood[col] + 10
# %%
l = Lasso(alpha=0.0000001).fit(X_tr, y_tr)
# l = LinearRegression().fit(X_tr, y_tr)
dummy = DummyRegressor().fit(X_tr, y_tr)
print("Dummry", mean_absolute_error(dummy.predict(X_te), y_te))
print("Model", mean_absolute_error(l.predict(X_te), y_te))
# %%
model = Boot(LinearRegression())
model.fit(X_tr, y_tr.target.values, n_boots=20)

# Predictions

preds, intervals = model.predict(X_ood, uncertainty=0.05, n_boots=20)
uncertainty = intervals[:, 1] - intervals[:, 0]
full = X_te.append(X_ood)
preds, intervals = model.predict(full, uncertainty=0.05, n_boots=20)
uncertainty = intervals[:, 1] - intervals[:, 0]
# %%

g = Lasso(alpha=0.001)
g.fit(full, uncertainty)
print("R2:", r2_score(g.predict(full), uncertainty))
print("Coeficients", g.coef_)

## Explanation
explainer = shap.LinearExplainer(g, full, feature_dependence="interventional")  #
shap_values = explainer(full)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
)
# %%

np.abs(exp).mean()
# %%
exp.head(1)
# %%
shap_values = explainer(pd.DataFrame([10, 10, 10]).T)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
)
# %%
np.abs(exp).mean()
# %%
s = g.coef_ * (np.array([10, 10, 10]) - np.mean(full).values)
# %%
plt.title("Local Shapley values for linear regression for synthetic data ")
plt.ylabel("Shaple Value")
plt.bar(["$\phi_1$", "$\phi_2$", "$\phi_3$"], s)
plt.savefig("experiments/results/analytical.png")
plt.show()

# %%
