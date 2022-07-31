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
from src.estimators import evaluate_doubt, evaluate_mapie, evaluate_nasa

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
