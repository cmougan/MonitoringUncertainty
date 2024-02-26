# %%
# Import candidate models
from sklearn.utils.validation import check_is_fitted
from doubt import Boot
from sklearn.linear_model import Lasso, LogisticRegression
from skshift import ExplanationShiftDetector
from mapie.regression import MapieRegressor
from sklearn.preprocessing import StandardScaler
from src.estimators import evaluate_nasa, evaluate_doubt, evaluate_mapie
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import random
import pdb

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
        uncertainty_m_res = []
        uncertainty_n_res = []
        ks_res = []
        psi_res = []
        esd_res = []
        target_shift = []
        for idx, col in enumerate(X.columns):
            values = defaultdict(list)

            # OOD
            X_ood = X_te.copy()
            X_ood[col] = np.linspace(-2, 4, X_te.shape[0])

            y_ood = (
                X_ood["Var1"] ** 2
                + X_ood["Var2"]
                + np.random.normal(0, 0.01, X_ood.shape[0])
            )

            X_tot = pd.concat([X_te, X_ood])
            y_tot = pd.concat([y_te, pd.Series(y_ood)]).reset_index(drop=True)
            a1 = np.zeros(len(y_te))
            a2 = np.ones(len(y_ood))
            y_tot_esd = np.concatenate([a1, a2])

            # Predictions
            base_model = base_regressor(**kwargs)
            base_model.fit(X_tr, y_tr)
            preds_tr = base_model.predict(X_tr)
            preds = base_model.predict(X_ood)
            ## Doubt
            _, intervals = evaluate_doubt(
                model=base_regressor(**kwargs),
                X_tr=X_tr,
                X_te=X_ood,
                y_tr=y_tr.target.values,
                y_te=y_ood,
                uncertainty=0.05,
                desaggregated=True,
            )

            # Mapie
            _, intervals_m = evaluate_mapie(
                model=base_regressor(**kwargs),
                X_tr=X_tr,
                X_te=X_ood,
                y_tr=y_tr.target.values,
                y_te=y_ood,
                uncertainty=[0.05],
                desaggregated=True,
            )
            # Nasa
            _, intervals_n = evaluate_nasa(
                model=base_regressor(**kwargs),
                X_tr=X_tr,
                X_te=X_ood,
                y_tr=y_tr.target.values,
                y_te=y_ood,
                uncertainty=0.05,
                desaggregated=True,
            )
            # Explanaition Shift
            esd_model = base_regressor(**kwargs)
            esd_model.fit(X_tr, y_tr)
            esd = ExplanationShiftDetector(
                model=esd_model,
                gmodel=LogisticRegression(),
                masker=True,
                data_masker=X_tr,
            )
            esd.fit_detector(X_tr, X_ood)
            preds_ood_esd = esd.predict_proba(X_ood)[:, 1]

            # Statistics
            df = pd.DataFrame(
                intervals[:, 1] - intervals[:, 0], columns=["uncertainty"]
            )
            df["uncertainty_m"] = intervals_m[:, 1] - intervals_m[:, 0]
            df["uncertainty_n"] = intervals_n[:, 1] - intervals_n[:, 0]
            df["error"] = np.abs(preds - y_ood.values)

            df["esd"] = np.abs(preds_ood_esd - np.ones(len(preds_ood_esd)))

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
            df = df.rename(
                columns={
                    "uncertainty_m": "Mapie",
                    "uncertainty_n": "Nasa",
                    "uncertainty": "Doubt",
                    "esd": "Explanation Shift",
                }
            )
            for index, col in enumerate(df.columns):
                values[col] = df[col]

            uncertainty_res.append(
                mean_absolute_error(values["error"], values["Doubt"])
            )
            uncertainty_m_res.append(
                mean_absolute_error(values["error"], values["Mapie"])
            )
            uncertainty_n_res.append(
                mean_absolute_error(values["error"], values["Nasa"])
            )
            ks_res.append(mean_absolute_error(values["error"], values["ks"]))
            psi_res.append(mean_absolute_error(values["error"], values["PSI"]))
            target_shift.append(
                mean_absolute_error(values["error"], values["target_shift"])
            )
            esd_res.append(
                mean_absolute_error(values["error"], values["Explanation Shift"])
            )

        resultados = pd.DataFrame(
            {
                "Doubt": uncertainty_res,
                "Mapie": uncertainty_m_res,
                "Nasa": uncertainty_n_res,
                "Explanation Shift": esd_res,
                "ks": ks_res,
                "psi": psi_res,
                "target_shift": target_shift,
            }
        )
        print("Data Synthetic")
        print(resultados.mean())
        resultados.loc["mean"] = resultados.mean()
        plot = False
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

a.to_csv("experiments/results/monitoring_synthetic.csv")

# %%
