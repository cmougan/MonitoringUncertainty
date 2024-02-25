import numpy as np
import random
from doubt import Boot
from mapie.regression import MapieRegressor
import pdb

# Initialise random number generator
rng = np.random.default_rng(4242)
random.seed(0)


def evaluate_nasa(model, X_tr, X_te, y_tr, y_te, uncertainty=0.05, desaggregated=False):
    n_boots = int(np.sqrt(len(X_tr)))
    # n_boots = 20

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
        X_btr = X_tr.values[train_idxs, :]
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
    if desaggregated:
        return bootstrap_preds, intervals

    coverage = np.mean((y_te > intervals[:, 0]) & (y_te < intervals[:, 1]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width


def evaluate_doubt(
    model, X_tr, X_te, y_tr, y_te, uncertainty=0.05, desaggregated=False
):
    n_boots = int(np.sqrt(len(X_tr)))
    # n_boots = 20

    bmodel = Boot(model, random_seed=4242)
    bmodel.fit(X_tr, y_tr, n_boots=n_boots)
    preds, intervals = bmodel.predict(X_te, uncertainty=uncertainty, n_boots=n_boots)

    if desaggregated:
        return preds, intervals
    coverage = np.mean((y_te > intervals[:, 0]) & (y_te < intervals[:, 1]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width


def evaluate_mapie(
    model, X_tr, X_te, y_tr, y_te, uncertainty=0.05, desaggregated=False
):
    n_boots = int(np.sqrt(len(X_tr)))
    # n_boots = 20
    bmodel = MapieRegressor(model, cv=n_boots)
    bmodel.fit(X_tr, y_tr)
    preds, intervals = bmodel.predict(X_te, alpha=uncertainty)

    if desaggregated:
        return preds, intervals

    coverage = np.mean((y_te > intervals[:, 0, 0]) & (y_te < intervals[:, 1, 0]))
    mean_width = np.mean(intervals[:, 1] - intervals[:, 0])
    return coverage, mean_width
