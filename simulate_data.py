"""
Simulate data about agricultural land values, to illustrate various concepts about
estimating model prediction error.

The generating distributions are loosely based on Texas Hill Country 2020 land values,
from https://www.recenter.tamu.edu/data/rural-land/.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


## Simulation parameters
np.random.seed(18)

n = 100

acreage_mean = 120
acreage_sd = 30

price_sd = 250000


## Generate data
df = pd.DataFrame({
    "acres": np.random.normal(acreage_mean, acreage_sd, n)
})

noise = np.random.normal(loc=0, scale=price_sd, size=n)
df["sq_acres"] = df["acres"]**2
df["price"] = 2000 * df["acres"] + 50 * df["sq_acres"] + noise


## Generic plot style
layout = dict(
    font=dict(family="Arial", size=32),
    template="simple_white",
)

marker_size = 28


## Plot the data
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["acres"], y=df["price"],
    mode="markers",
    marker=dict(
        symbol="circle",
        color="rgba(100, 149, 237, 0.35)",
        size=marker_size,
        line=dict(width=2, color="#15388d"),
    ),
    showlegend=False
))

fig.update_layout(layout)
fig.update_layout(
    xaxis_title = "Acres",
    yaxis_title = "Sale price"
)

fig.write_image("sim_farm_sales.png", height=1400, width=1400)


## K-fold cross-validation to choose the best model form
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

linear_rmse = []
quadratic_rmse = []

target = "price"

xgrid = pd.DataFrame({
    "acres": np.linspace(df["acres"].min() - 5, df["acres"].max() + 5, 100)
})
xgrid["sq_acres"] = xgrid["acres"]**2

kfold = KFold(n_splits=5)
for ix_train, ix_test in kfold.split(df):
    df_train = df.loc[ix_train]
    df_test = df.loc[ix_test]

    model_a = linear_model.LinearRegression()
    model_a.fit(df_train[["acres"]], df_train[[target]])
    linear_rmse.append(mse(df_test[target], model_a.predict(df_test[["acres"]]))**0.5)

    model_b = linear_model.LinearRegression()
    model_b.fit(df_train[["acres", "sq_acres"]], df_train[target])
    quadratic_rmse.append(mse(df_test[target], model_b.predict(df_test[["acres", "sq_acres"]]))**0.5)

    fig.add_trace(go.Scatter(
        x=xgrid["acres"], y=model_a.predict(xgrid[["acres"]]).flatten(),
        mode="lines",
        line=dict(width=2, dash="dash", color="orange"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=xgrid["acres"], y=model_b.predict(xgrid[["acres", "sq_acres"]]).flatten(),
        mode="lines",
        line=dict(width=2, dash="dash", color="purple"),
        showlegend=False
    ))

print(f"Linear model CV RMSE average: {np.array(linear_rmse).mean()}")
print(f"Quadratic model CV RMSE average: {np.array(quadratic_rmse).mean()}")

fig.add_annotation(x=205, y=1.7e6,
            text="K-fold CV linear models<br>RMSE: $240,000",
            showarrow=False,
            font=dict(color="orange"))

fig.add_annotation(x=185, y=2.8e6,
            text="K-fold CV quadratic models<br>RMSE: $228,000",
            showarrow=False,
            font=dict(color="purple"))

fig.write_image("cv_model_fits.png", height=1400, width=1400)
