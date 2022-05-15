# %%
# Import external libraries
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_theme(style="whitegrid")
plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})
# %%
df = pd.read_csv("experiments/computationalPerformance.csv")
# %%
rows = {
    "Airfoil": 1503,
    "Concrete": 1030,
    "FishToxicity": 908,
    "ForestFire": 517,
    "NewTaipeiHousing": 414,
    "PowerPlant": 9568,
    "Protein": 45730,
    "Servo": 167,
}
# %%
df["rows"] = df.Data.map(rows)
# %%
for model in df.Model.unique():
    ## Params
    if model == "LinearRegression":
        legend = True
    else:
        legend = False

    aux = df[df.Model == model]  # .groupby('Data').mean()
    aux = aux.sort_values("rows")
    aux = aux.drop(columns=["Uncertainty", "rows"])
    aux = aux.melt(id_vars=["Data", "Model"], value_vars=["Nasa", "Doubt", "Mapie"])
    g = sns.catplot(
        data=aux,
        kind="bar",
        x="Data",
        y="value",
        hue="variable",
        ci="sd",
        palette="dark",
        alpha=0.6,
        height=6,
        legend=False,
    )
    g.despine(left=True)
    g.set_xticklabels(rotation=30)
    g.set_axis_labels("", "Wall Time (s)")
    g.fig.suptitle(model)
    if legend:
        g.add_legend(bbox_to_anchor=(0.3, 0.7))
    g.savefig("experiments/results/computationalPerformance" + model + ".png")

# %%
