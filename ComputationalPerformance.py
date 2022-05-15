# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from doubt import Boot
from doubt.datasets import (
    Airfoil,
    Blog,
    Concrete,
    CPU,
    FacebookComments,
    FishBioconcentration,
    FishToxicity,
    ForestFire,
    NewTaipeiHousing,
    PowerPlant,
    Protein,
    Servo,
    SpaceShuttle,
)
from mapie.regression import MapieRegressor
import time

# %%
# ## Setting up
def evaluate_nasa(model, X_tr, X_te, y_tr, y_te, uncertainty=0.05):
    n_boots = int(np.sqrt(len(X_tr)))

    # Calculate training residuals
    model.fit(X_tr, y_tr)
    tr_preds = model.predict(X_tr)
    te_preds = model.predict(X_te)
    tr_residuals = y_tr - tr_preds

    n_train = X_tr.shape[0]
    n_test = X_te.shape[0]

    # Initialise random number generator
    rng = np.random.default_rng(4242)

    # Compute the model variances
    bootstrap_preds = np.empty((n_boots, n_test))
    for boot_idx in range(n_boots):
        train_idxs = rng.choice(range(n_train), size=n_train, replace=True)
        X_btr = X_tr[train_idxs, :]
        y_btr = y_tr[train_idxs]

        model.fit(X_btr, y_btr)

        bootstrap_pred = model.predict(X_te)
        bootstrap_preds[boot_idx] = bootstrap_pred

    # Centre the bootstrapped predictions across the bootstrap dimension
    bootstrap_preds = np.mean(bootstrap_preds, axis=0) - bootstrap_preds

    # Add up the bootstrap predictions and the hybrid train/val residuals
    C = np.array([m + o for m in bootstrap_preds for o in tr_residuals])

    # Calculate the intervals
    intervals = np.expand_dims(te_preds, -1) + np.transpose(
        np.quantile(C, q=[uncertainty / 2, 1 - uncertainty / 2], axis=0)
    )

    coverage = np.mean((y_te > intervals[:, 0]) & (y_te < intervals[:, 1]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width


# %%
def evaluate_doubt(model, X_tr, X_te, y_tr, y_te, uncertainty=0.05):
    n_boots = int(np.sqrt(len(X_tr)))

    bmodel = Boot(model, random_seed=4242)
    bmodel.fit(X_tr, y_tr, n_boots=n_boots)
    preds, intervals = bmodel.predict(X_te, uncertainty=uncertainty, n_boots=n_boots)

    coverage = np.mean((y_te > intervals[:, 0]) & (y_te < intervals[:, 1]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width


# %%
def evaluate_mapie(model, X_tr, X_te, y_tr, y_te, uncertainty=0.05):
    bmodel = MapieRegressor(model)
    bmodel.fit(X_tr, y_tr)
    preds, intervals = bmodel.predict(X_te, alpha=uncertainty)

    coverage = np.mean((y_te > intervals[:, 0, 0]) & (y_te < intervals[:, 1, 0]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width


# %%
## Scaling
datasets = []
scaler = StandardScaler()

# Add Doubt datasets
dataset_classes = [
    Airfoil,
    Concrete,
    FishToxicity,
    ForestFire,
    NewTaipeiHousing,
    PowerPlant,
    Protein,
    Servo,
]

for dataset_class in dataset_classes:
    dataset = dataset_class()
    dataset._data = dataset._data.sample(n=min(len(dataset), 2000), random_state=4242)
    X_tr, X_te, y_tr, y_te = dataset.split(test_size=0.1, random_seed=4242)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    datasets.append((dataset_class.__name__, X_tr, X_te, y_tr, y_te))

len(datasets)


for name, X_tr, X_te, y_tr, y_te in datasets:
    print(f"{name}: {len(X_tr) + len(X_te):,} samples, {X_tr.shape[-1]:,} features")

# %%
# ## Models
models = [LinearRegression(), DecisionTreeRegressor(), XGBRegressor()]
results = []

for model in models:
    for dataset in tqdm(datasets):
        for uncertainty in tqdm(np.arange(0.01, 0.26, 0.01), leave=False):
            step1 = time.time()
            nasa_coverage, nasa_mean_width = evaluate_nasa(
                model, *dataset[1:], uncertainty=uncertainty
            )
            step2 = time.time()
            doubt_coverage, doubt_mean_width = evaluate_doubt(
                model, *dataset[1:], uncertainty=uncertainty
            )
            step3 = time.time()
            mapie_coverage, mapie_mean_width = evaluate_mapie(
                model, *dataset[1:], uncertainty=uncertainty
            )
            step4 = time.time()
            results.append(
                [
                    dataset[0],
                    type(model).__name__,
                    uncertainty,
                    np.abs(step2 - step1),
                    np.abs(step3 - step2),
                    np.abs(step4 - step3),
                ]
            )

# %%
# Save results
dt = pd.DataFrame(
    results, columns=["Data", "Model", "Uncertainty", "Nasa", "Doubt", "Mapie"]
)
dt.to_csv("experiments/computationalPerformance.csv", index=False)
