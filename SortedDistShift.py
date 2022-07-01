# %%
# Import candidate models
from doubt import Boot
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Import datasets
from doubt.datasets import (
    Airfoil,
    BikeSharingDaily,
    BikeSharingHourly,
    Blog,
    Concrete,
    CPU,
    FacebookComments,
    FacebookMetrics,
    FishBioconcentration,
    FishToxicity,
    ForestFire,
    GasTurbine,
    Nanotube,
    NewTaipeiHousing,
    Parkinsons,
    PowerPlant,
    Protein,
    Servo,
    SolarFlare,
    SpaceShuttle,
    Stocks,
    Superconductivity,
    TehranHousing,
    Yacht,
)

# Import external libraries
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from mapie.regression import MapieRegressor

plt.style.use("seaborn-whitegrid")


import warnings
from collections import defaultdict

from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# Import internal classes
from src.psi import psi
from tqdm.notebook import tqdm

from xgboost import XGBRegressor


# %%
dataset_classes = [
    # Airfoil,
    # Concrete,
    # ForestFire,
    # Parkinsons,
    # PowerPlant,
    Protein,
    BikeSharingHourly,
    FishToxicity,
    Nanotube,
]
for dataset in dataset_classes:
    print(dataset.__name__, dataset().shape)


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
        axs[num_rows - 1, j].set_xlabel("Sorted column index")
    for i in range(num_rows):
        axs[i, 0].set_ylabel("Normalised values")
    return fig, axs


# %%


def kol_smi(x):
    return ks_2samp(x, BASE_COMP).statistic


def kol_smi_preds(x):
    return ks_2samp(x, BASE_COMP_PREDS).statistic


def psi_stat(x):
    return psi(x, BASE_COMP)


# In[18]:


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

        # Some datasets have multilabel, we stick to the first one.
        try:
            data["target"] = y
        except:
            data["target"] = y[:, 0]

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
        uncertainty_m_res = []
        ks_res = []
        psi_res = []
        target_shift = []
        for idx, col in tqdm(
            enumerate(X.columns), total=len(X.columns), desc="Columns"
        ):
            if idx > 9:  # Limit in columns
                continue
            values = defaultdict(list)

            # Sort data on the column
            data = data.sort_values(col).reset_index(drop=True).copy()

            # Train Test Split
            data_sub = data.iloc[:oneThird]
            data_train = data.iloc[oneThird:twoThird]
            data_up = data.iloc[twoThird:]

            X_tot = data.drop(columns="target")
            X_tr = data_train.drop(columns="target")
            X_sub = data_sub.drop(columns="target")
            X_up = data_up.drop(columns="target")

            y_tot = data[["target"]].target.values
            y_tr = data_train[["target"]].target.values
            y_sub = data_sub[["target"]].target.values
            y_up = data_up[["target"]].target.values

            # Fit the regressor
            regressor = Boot(base_regressor(**kwargs))
            regressor.fit(X_tr, y_tr, n_boots=20)

            # Predictions
            preds_tr = regressor.predict(X_tr)
            preds, intervals = regressor.predict(
                X_tot, uncertainty=0.05, n_boots=n_boots
            )

            mapie = MapieRegressor(base_regressor(**kwargs))

            mapie.fit(X_tr, y_tr)
            preds_m, intervals_m = mapie.predict(X_tot, alpha=[0.05])

            # Statistics
            df = pd.DataFrame(
                intervals[:, 1] - intervals[:, 0], columns=["uncertainty"]
            )

            df["uncertainty_m"] = intervals_m[:, 1] - intervals_m[:, 0]

            df["error"] = np.abs(preds - y_tot)

            ### KS Test
            df["ks"] = data[col]
            global BASE_COMP
            BASE_COMP = data[col]
            df[["ks"]] = (
                df[["ks"]].rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5)).apply(kol_smi)
            )  # Takes ages
            ### PSI Test
            df["PSI"] = data[col]
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
            )  # Takes ages

            ### Rolling window on all
            df[df.columns[~df.columns.isin(["ks", "PSI", "target_shift"])]] = (
                df[df.columns[~df.columns.isin(["ks", "PSI", "target_shift"])]]
                .rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5))
                .mean()
            ).dropna()

            ## Scaling
            df = df.dropna()
            df = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

            # Convert to dic for plotting
            for index, col in enumerate(df.columns):
                values[col] = df[col]

            uncertainty_res.append(
                mean_absolute_error(values["error"], values["uncertainty"])
            )
            uncertainty_m_res.append(
                mean_absolute_error(values["error"], values["uncertainty_m"])
            )
            ks_res.append(mean_absolute_error(values["error"], values["ks"]))
            psi_res.append(mean_absolute_error(values["error"], values["PSI"]))
            target_shift.append(
                mean_absolute_error(values["error"], values["target_shift"])
            )

            # Plotting
            if plot:
                for name, vals in values.items():
                    if idx == 0:
                        axs[idx // 3, idx % 3].plot(vals, label=f"{name} values")
                    else:
                        axs[idx // 3, idx % 3].plot(vals)
        resultados = pd.DataFrame(
            {
                "uncertainty": uncertainty_res,
                "uncertainty_m": uncertainty_m_res,
                "ks": ks_res,
                "psi": psi_res,
                "target_shift": target_shift,
            }
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
print("Linear Regression")
for dataset in dataset_classes:
    monitoring_plot(dataset, LinearRegression)
# %%
print("Poisson Regressor")
for dataset in dataset_classes:
    monitoring_plot(dataset, PoissonRegressor)
# %%
print("Decision Tree Regressor Depth 10")
for dataset in dataset_classes:
    monitoring_plot(dataset, DecisionTreeRegressor, max_depth=10)
# %%
print("Random Forest Regressor")
for dataset in dataset_classes:
    monitoring_plot(dataset, RandomForestRegressor, n_estimators=20)
# %%
print("XGBoost Regressor")
for dataset in dataset_classes:
    monitoring_plot(dataset, XGBRegressor)
# %%
print("MLP Regressor")
for dataset in dataset_classes:
    monitoring_plot(dataset, MLPRegressor)

# %%
"""
# ## Rolling window experiment
dfs = {}
for dataset in dataset_classes:
    uncertainty_loop = []
    ks_loop = []
    psi_loop = []
    params = np.array(range(20, 400, 20))
    for rs in params:
        if dataset().shape[0] / 3 > rs:
            print(rs)
            res = monitoring_plot(
                dataset, LinearRegression, ROLLING_STAT=rs, plot=False
            )
            uncertainty_loop.append(res.loc["mean"][0])
            ks_loop.append(res.loc["mean"][1])
            psi_loop.append(res.loc["mean"][2])

    agg = pd.DataFrame([uncertainty_loop, ks_loop, psi_loop, params]).T
    agg.columns = ["uncertainty", "ks", "psi", "parameters"]
    dfs[dataset.__name__] = agg



fig, axs = plt.subplots(1, len(dfs))
fig.suptitle(
    "Impact of rolling window size over unsupervised model monitoring techniques"
)
for idx, key in enumerate(dfs.keys()):
    axs[idx].title.set_text("Dataset:{}".format(key))
    dt = dfs[key]
    dt["params"] = range(dt.shape[0])
    axs[idx].plot(dt["params"], dt["uncertainty"].values, label="Uncertainty")
    axs[idx].plot(dt["params"], dt["ks"].values, label="K-S")
    axs[idx].plot(dt["params"], dt["psi"].values, label="PSI")
    if idx == 2:
        axs[idx].legend(bbox_to_anchor=(1, 1))


plt.show()
"""
