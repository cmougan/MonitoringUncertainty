# %%
# Import candidate models

from sklearn.utils.validation import check_is_fitted
from doubt import Boot
from sklearn.linear_model import (
    Lasso,
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import random

random.seed(0)

# Import external libraries
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import warnings
from collections import defaultdict

from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 16})

# Import internal classes
from distributions import DistributionShift
from src.psi import psi
from tqdm.notebook import tqdm
from xgboost import XGBRegressor
from tabulate import tabulate


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


def kol_smi(x):
    return ks_2samp(x, BASE_COMP).statistic


def kol_smi_preds(x):
    return ks_2samp(x, BASE_COMP_PREDS).statistic


def psi_stat(x):
    return psi(x, BASE_COMP)


def monitoring_plot(
    dataset,
    base_regressor: type,
    n_boots: int = 20,
    ROLLING_STAT: int = 50,
    plot: bool = True,
    **kwargs,
):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standard_scaler = StandardScaler()
        # Split data

        X = dataset.drop(columns="target")
        y = dataset[["target"]]

        # Train test splitting points
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.5)

        # Fit the regressor
        regressor = Boot(base_regressor(**kwargs))
        regressor.fit(X_tr, y_tr.target.values, n_boots=20)

        # Initialize plots

        fig, axs = plt.subplots(
            1,
            3,
            sharex=True,
            sharey=True,
            figsize=(14, 3.5),
        )
        # fig.suptitle(f"Monitoring plot for synthetic data under feature drift with {base_regressor.__name__}",fontsize=16,)

        uncertainty_res = []
        ks_res = []
        psi_res = []
        target_shift = []
        for idx, col in tqdm(enumerate(X.columns), total=len(X.columns)):
            values = defaultdict(list)

            # OOD
            X_ood = X_te.copy()
            X_ood[col] = np.linspace(-2, 4, X_te.shape[0])

            y_ood = (
                X_ood["Var1"] ** 2
                + X_ood["Var2"]
                + np.random.normal(0, 0.01, X_ood.shape[0])
            )

            # Predictions
            preds_tr = regressor.predict(X_tr)
            preds, intervals = regressor.predict(
                X_ood, uncertainty=0.05, n_boots=n_boots
            )

            # Statistics
            df = pd.DataFrame(
                intervals[:, 1] - intervals[:, 0], columns=["uncertainty"]
            )
            df["error"] = np.abs(preds - y_ood.values)

            ### KS Test
            df["ks"] = X_ood[col].values
            global BASE_COMP
            BASE_COMP = X_tr[col].values
            df[["ks"]] = (
                df[["ks"]].rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5)).apply(kol_smi)
            )  # Takes ages
            ### PSI Test
            df["PSI"] = X_ood[col].values
            df[["PSI"]] = (
                df[["PSI"]]
                .rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5))
                .apply(psi_stat)
            )  # Takes ages
            # Label Shift
            global BASE_COMP_PREDS
            BASE_COMP_PREDS = preds_tr
            df["target_shift"] = preds
            df[["target_shift"]] = (
                df[["target_shift"]]
                .rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5))
                .apply(kol_smi_preds)
            )

            ### Rolling window on all
            df[df.columns[~df.columns.isin(["ks", "PSI", "target_shift"])]] = (
                df[df.columns[~df.columns.isin(["ks", "PSI", "target_shift"])]]
                .rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5))
                .mean()
            ).dropna()

            ## Scaling
            df = df.dropna()
            try:
                check_is_fitted(standard_scaler)
                df = pd.DataFrame(standard_scaler.transform(df), columns=df.columns)
            except:
                standard_scaler.fit(df)
                df = pd.DataFrame(standard_scaler.transform(df), columns=df.columns)

            # Convert to dic for plotting
            for index, col in enumerate(df.columns):
                values[col] = df[col]

            uncertainty_res.append(
                mean_absolute_error(values["error"], values["uncertainty"])
            )
            ks_res.append(mean_absolute_error(values["error"], values["ks"]))
            psi_res.append(mean_absolute_error(values["error"], values["PSI"]))
            target_shift.append(
                mean_absolute_error(values["error"], values["target_shift"])
            )

            # Plotting

            for name, vals in values.items():
                axs[idx].plot(vals, label=f"{name} values")

        resultados = pd.DataFrame(
            {
                "uncertainy": uncertainty_res,
                "ks": ks_res,
                "psi": psi_res,
                "target_shift": target_shift,
            }
        )
        print("Data Synthetic")
        print(resultados.mean())
        resultados.loc["mean"] = resultados.mean()

        if plot:
            plt.legend(loc=2, prop={"size": 6})
            axs[0].set_title("Quadratic feature")
            axs[1].set_title("Linear feature")
            axs[2].set_title("Random feature")
            plt.savefig("experiments/results/syntheticDegradation.eps", format="eps")
            plt.savefig("experiments/results/syntheticDegradation.png")
            plt.show()
        return resultados


# %%
a = monitoring_plot(df, Lasso, alpha=0.00001)

# %%
a

# %%
