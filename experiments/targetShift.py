from sklearn.linear_model import LinearRegression
from doubt import Boot
import seaborn as sns
from doubt import QuantileRegressionForest as QRF

import pandas as pd
import os
import matplotlib.pyplot as plt
from doubt.datasets import PowerPlant


path_out = os.path.join("experiments", "results", "targetShift")

datasets = {"PowerPlant": PowerPlant().split()}
models = {"Linear": Boot(LinearRegression()), "QuantileRegressor": QRF()}


for dataset in datasets:
    for model in models:
        X, y = datasets[dataset]
        clf = models[model]

        # Join and order Dataframe
        columns = [f"col_{num}" for num in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)
        X["target"] = y
        X = X.sort_values("target").reset_index(drop=True)

        # Determine fraction to use in test (will be x2 the split point)
        split_point = int(X.shape[0] * 0.35)

        # Select smaller and bigger than splitting point
        train = X.iloc[split_point : X.shape[0] - split_point]
        pre = X.iloc[:split_point]
        post = X.iloc[X.shape[0] - split_point :]
        test = pre.append(post)

        # Train Test Split
        X_tr = train.drop(columns="target")
        y_tr = train["target"].values

        X_te = test.drop(columns="target")
        y_te = test["target"].values

        clf = Boot(LinearRegression())

        clf.fit(X_tr, y_tr)

        preds = clf.predict(
            X.sort_values("target").drop(columns=["target"]), uncertainty=0.05
        )

        p = preds[1][:, 1] - preds[1][:, 0]

        plt.figure()
        plt.plot(p)
        plt.vlines(split_point, ymin=p.min(), ymax=p.max(), colors="k")
        plt.vlines(X.shape[0] - split_point, ymin=p.min(), ymax=p.max(), colors="k")
        file_name = str(model) + "_" + str(dataset) + "_" + "full.png"
        fig_path = os.path.join(path_out, file_name)
        plt.savefig(fig_path)
        plt.close()

        pred_test = clf.predict(X_te, uncertainty=0.05)
        p_test = pred_test[1][:, 1] - pred_test[1][:, 0]

        pred_train = clf.predict(X_tr, uncertainty=0.05)
        p_train = pred_train[1][:, 1] - pred_train[1][:, 0]

        plt.figure()
        sns.kdeplot(p_train, label="train")
        sns.kdeplot(p_test, label="test")
        plt.legend()
        file_name = str(model) + "_" + str(dataset) + "_" + "kde.png"
        fig_path = os.path.join(path_out, file_name)
        plt.savefig(fig_path)
        plt.close()
