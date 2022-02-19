# %%
# Import candidate models
from doubt import Boot
from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Import datasets
from doubt.datasets import (
    Airfoil,
    Concrete,
)

# Import external libraries
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import warnings
from collections import defaultdict

get_ipython().run_line_magic("matplotlib", "inline")
from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

# Import internal classes
from src.psi import psi
from tqdm.notebook import tqdm
from xgboost import XGBRegressor
import shap

# %%
dataset_classes = [
    Airfoil,
    Concrete,
    # FishToxicity,
    # ForestFire,
    # NewTaipeiHousing,
    # PowerPlant,
    # Protein,
    # Servo,
]
for dataset in dataset_classes:
    print(dataset().shape)


# %%


def initialise_plot(num_rows: int, num_cols: int, base_regressor: type, dataset):
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        figsize=(14, 3.5 * num_rows),
    )
    fig.suptitle(
        f"Monitoring plot for the {dataset.__name__} dataset under feature drift with {base_regressor.__name__}",
        fontsize=16,
    )
    for j in range(num_cols):
        axs[num_rows - 1, j].set_xlabel("Sorted columnd index")
    for i in range(num_rows):
        axs[i, 0].set_ylabel("Normalised values")
    return fig, axs


def kol_smi(x):
    return ks_2samp(x, BASE_COMP).statistic


def psi_stat(x):
    return psi(x, BASE_COMP)


# In[15]:


def monitoring_plot(
    dataset,
    base_regressor: type,
    n_boots: int = 20,
    ROLLING_STAT: int = 100,
    plot: bool = True,
    **kwargs,
):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initialise the scaler
        standard_scaler = StandardScaler()

        # Load the dataset and split it
        X, _, y, _ = dataset().split(test_size=0.001, random_seed=4242)

        # Scale the dataset
        X = standard_scaler.fit_transform(X)

        # Back to dataframe
        X = pd.DataFrame(X, columns=["Var %d" % (i + 1) for i in range(X.shape[1])])
        data = X.copy()
        data["target"] = y

        # Train test splitting points
        fracc = 0.33
        oneThird = int(data.shape[0] * fracc)
        twoThird = data.shape[0] - int(data.shape[0] * fracc)

        # Initialize plots
        num_rows = X.shape[1] // 3
        if X.shape[1] % 3 > 0:
            num_rows += 1
        fig, axs = initialise_plot(
            num_rows=num_rows,
            num_cols=3,
            base_regressor=base_regressor,
            dataset=dataset,
        )

        uncertainty_res = []
        ks_res = []
        shap_res = []
        for idx, col in tqdm(enumerate(X.columns), total=len(X.columns)):
            values = defaultdict(list)

            # Sort data on the column
            data = data.sort_values(col).reset_index(drop=True).copy()

            # Train Test Split
            data_sub = data.iloc[:oneThird]
            data_train = data.iloc[oneThird:twoThird]
            data_up = data.iloc[twoThird:]

            X_tot = data.drop(columns="target")
            X_tr = data_train.drop(columns="target")

            y_tot = data[["target"]].target.values
            y_tr = data_train[["target"]].target.values

            # Shap
            # Fit the regressor
            regressor = base_regressor()
            regressor.fit(X_tr, y_tr)

            # Predictions
            preds = regressor.predict(X_tot)
            # use Decision Tree specific SHAP to explain
            explainer = shap.Explainer(regressor)
            shap_values = explainer(X_tot)
            shap_values = pd.DataFrame(data=shap_values.values, columns=X_tot.columns)


            # shap_res.append(np.mean(df.shap_diff.values))
            # Fit the regressor
            regressor = Boot(base_regressor(**kwargs))
            regressor.fit(X_tr, y_tr, n_boots=20)

            # Predictions
            preds, intervals = regressor.predict(
                X_tot, uncertainty=0.05, n_boots=n_boots
            )

            # Statistics
            df = pd.DataFrame(
                intervals[:, 1] - intervals[:, 0], columns=["uncertainty"]
            )
            df["error"] = np.abs(preds - y_tot)
            ## TODO: NOT SURE ABOUT THIS ABSOLUTE VALUE THINGY
            df["mean_shap"] = np.mean(np.abs(shap_values), axis=1)

            ### KS Test
            df["ks"] = data[col]
            global BASE_COMP
            BASE_COMP = data[col]
            df[["ks"]] = (
                df[["ks"]].rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5)).apply(kol_smi)
            )  # Takes ages
            
            ### Rolling window on all
            df[df.columns[~df.columns.isin(["ks"])]] = (
                df[df.columns[~df.columns.isin(["ks"])]]
                .rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5))
                .mean()
            ).dropna()

            ## Scaling
            # TODO: Fit scaler on trainning data
            df = df.dropna()
            df = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

            # Convert to dic for plotting
            for index, col in enumerate(df.columns):
                values[col] = df[col]

            uncertainty_res.append(
                mean_absolute_error(values["error"], values["uncertainty"])
            )
            ks_res.append(mean_absolute_error(values["error"], values["ks"]))
            shap_res.append(mean_absolute_error(values["error"], values["mean_shap"]))

            # Plotting
            if plot:
                for name, vals in values.items():
                    if idx == 0:
                        axs[idx // 3, idx % 3].plot(vals, label=f"{name} values")
                    else:
                        axs[idx // 3, idx % 3].plot(vals)
        resultados = pd.DataFrame(
            {"uncertainy": uncertainty_res, "ks": ks_res, "shap": shap_res}
        )
        print("Data", dataset.__name__)
        print(resultados.mean())
        resultados.loc["mean"] = resultados.mean()

        if plot:
            fig.legend()
            plt.savefig("fig.png")
            plt.show()
        return resultados
# %%

for dataset in dataset_classes:
    monitoring_plot(dataset, XGBRegressor)

# %%
1
# %%
