
import os
import numpy as np
import pandas as pd
import warnings
import itertools
import random
import statsmodels.api as sm
# porphet by Facebook
from fbprophet import Prophet
# timeseries analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
# data visual
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
root_path = "."
data_path = os.path.join(root_path, "data")


# # 1.read data and data information


df = pd.read_excel(os.path.join(data_path, "Groceries_Sales_data.xlsx"), parse_dates = [0])


# ## 1.1 Information of the timeseries


print('-'*60)
print('*** Head of the dataframe ***')
print('-'*60)
print(df.head())
print('-'*60)
print('*** Tail of the dataframe ***')
print('-'*60)
print(df.tail())
print('-'*60)
print('*** Information of the dataframe ***')
print('-'*60)
print(df.info())


# ## 1.2 Plot the timeseries data


def timeseries_plot(df, x, y):
    fig, ax = plt.subplots(figsize = (20, 7))
    chart = sns.lineplot(x = x, y = y, data = df, label = y)
    chart.set_title("%s Timeseries Data" % y)
    plt.legend()
    plt.show()

timeseries_plot(df, "Date", "Sales")


# # 2.Exploratory Data Analysis


# ## 2.1 Generate Date features and Exploratory


def date_features(df, dt, label = None):
    df = df.copy()
    df["date"] = df[dt]
    df["year"] = df["date"].dt.strftime("%Y")
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.strftime("%B")
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.strftime("%A")
    df["weekofyear"] = df["date"].dt.weekofyear
    
    X = df[[
        "year",
        "quarter",
        "month",
        "dayofyear",
        "dayofmonth",
        "dayofweek",
        "weekofyear"
    ]]
    if label:
        y = df[label]
        return X, y
    
    return X

X, y = date_features(df, dt = "Date", label = "Sales")
df_new = pd.concat([X, y], axis = 1)
df_new.head()


def bar_plot(df, x, y, categorical):
    fig, ax = plt.subplots(figsize = (20, 5))
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x = "month", y = "Sales", hue = "year", data = df_new)
    a.set_title("Store %s Data" % y, fontsize = 15)
    plt.legend(loc = "upper right")
    plt.show()

bar_plot(df_new, "month", "Sales", "year")


def bar_plots(df, x, y, xlabel, ylabel, nrows):
    fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=nrows)
    fig.set_size_inches(20,30)
    for ax in [ax1,ax2,ax3,ax4]:
        sns.barplot(data = df, x = x,y = y, ax = ax)
        ax.set(xlabel = xlabel, ylabel = ylabel)
        ax.set_title(title, fontsize=15)


fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
fig.set_size_inches(20,30)

monthAggregated = pd.DataFrame(df_new.groupby("month")["Sales"].sum()).reset_index().sort_values('Sales')
sns.barplot(data=monthAggregated,x="month",y="Sales",ax=ax1)
ax1.set(xlabel='Month', ylabel='Total Sales received')
ax1.set_title("Total Sales received By Month",fontsize=15)

monthAggregated = pd.DataFrame(df_new.groupby("dayofweek")["Sales"].sum()).reset_index().sort_values('Sales')
sns.barplot(data=monthAggregated,x="dayofweek",y="Sales",ax=ax2)
ax2.set(xlabel='dayofweek', ylabel='Total Sales received')
ax2.set_title("Total Sales received By Weekday",fontsize=15)

monthAggregated = pd.DataFrame(df_new.groupby("quarter")["Sales"].sum()).reset_index().sort_values('Sales')
sns.barplot(data=monthAggregated,x="quarter",y="Sales",ax=ax3)
ax3.set(xlabel='Quarter', ylabel='Total Sales received')
ax3.set_title("Total Sales received By Quarter",fontsize=15)

monthAggregated = pd.DataFrame(df_new.groupby("year")["Sales"].sum()).reset_index().sort_values('Sales')
sns.barplot(data=monthAggregated,x="year",y="Sales",ax=ax4)
ax4.set(xlabel='year', ylabel='Total Sales received')
ax4.set_title("Total Sales received By year",fontsize=15)


# # 3.Simple Prophet Model Training


# ## 3.1 Model Performance Metric


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ## 3.2 Split timeseries into Train and Test


df = df.rename(columns = {
    "Date": "ds",
    "Sales": "y"
})

end_date = "2019-12-31"
mask1 = (df["ds"] <= end_date)
mask2 = (df["ds"] > end_date)
X_train = df.loc[mask1]
X_test = df.loc[mask2]
print(X_train.shape)
print(X_test.shape)


pd.plotting.register_matplotlib_converters()
fig, ax = plt.subplots(figsize = (20, 5))
# X_train.plot(kind = "line", x = "ds", y = "y", color = "blue", label = "Train", ax = ax)
# X_test.plot(kind = "line", x = "ds", y = "y", color = "red", label = "Test", ax = ax)
sns.lineplot(x = "ds", y = "y", data = X_train, label = "Train")
sns.lineplot(x = "ds", y = "y", data = X_test, label = "Test")
plt.title("Sales Amount Training and Test data")
plt.legend()
plt.show()


# ## 3.3 Simple Prophet Model


# ### 3.3.1 Model training on train dataset and evaluate on test dataset


# model training and validation
model =  Prophet()
model.fit(X_train)
future = model.make_future_dataframe(periods = 57, freq = "D")
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7))
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))


# Plot the components of the model
fig = model.plot_components(forecast)


# plot the forecast
fig, ax = plt.subplots(1, figsize = (15, 5))
fig = model.plot(forecast, ax = ax)
plt.show()


# ### 3.3.2 Model performance on test dataset


X_test_forecast = model.predict(X_test)
print(X_test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7))
print(X_test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))


# plot the forecast with the actuals
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_forecast, ax = ax)
plot = plt.suptitle("Forecast vs Actuals")
plt.show()


# Plot the forecast with the actuals of 2020
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_forecast, ax = ax)
ax.set_xbound(lower = "2020-01-01", upper = "2020-02-26")
ax.set_ylim(0, 60000)
plot = plt.suptitle("First Week of January Forecast vs Actuals")
plt.show()


# Compare the Sales and forecasted Sales
fig, ax = plt.subplots(1, figsize = (15, 5))
X_test.plot(kind = "line", x = "ds", y = "y", color = "red", label = "Test", ax = ax)
X_test_forecast.plot(kind = "line", x = "ds", y = "yhat", color = "green", label = "Test Forecast", ax = ax)
plt.title("February 2020 Forecast vs Actuals")
plt.show()


# Model performance on X_test
mape = mean_absolute_percentage_error(X_test["y"], X_test_forecast["yhat"])
print("MAPE:%s" % round(mape, 4))


# ### 3.3.3 Adding holidays to the model


import holidays

holiday = pd.DataFrame([])
for date, name in sorted(holidays.UnitedStates(years = [2018, 2019, 2020]).items()):
    holiday_df = pd.DataFrame({
        "ds": date, 
        "holiday": "US-Hoildays"
    }, index = [0])
    holiday = holiday.append(holiday_df, ignore_index = True)

holiday["ds"] = pd.to_datetime(holiday["ds"], format = "%Y-%m-%d", errors = "ignore")
holiday


# Setup and train model with holidays
model_with_holidays = Prophet(holidays = holiday)
model_with_holidays.fit(X_test)
future = model_with_holidays.make_future_dataframe(periods = 57, freq = "D")
forecast = model_with_holidays.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7))
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))


# plot the forecast
fig, ax = plt.subplots(1, figsize = (15, 5))
fig = model_with_holidays.plot(forecast, ax = ax)
plt.show()


# Plot the components of the model
fig = model_with_holidays.plot_components(forecast)


X_test_forecast_holidays = model_with_holidays.predict(X_test)
print(X_test_forecast_holidays[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7))
print(X_test_forecast_holidays[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))


# plot the forecast with the actuals
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_forecast_holidays, ax = ax)
plot = plt.suptitle("Forecast vs Actuals")
plt.show()


# Plot the forecast with the actuals of 2020
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_forecast_holidays, ax = ax)
ax.set_xbound(lower = "2020-01-01", upper = "2020-02-26")
ax.set_ylim(0, 60000)
plot = plt.suptitle("First Week of January Forecast vs Actuals")
plt.show()


# Compare the Sales and forecasted Sales
fig, ax = plt.subplots(1, figsize = (15, 5))
X_test.plot(kind = "line", x = "ds", y = "y", color = "red", label = "Test", ax = ax)
X_test_forecast_holidays.plot(kind = "line", x = "ds", y = "yhat", color = "green", label = "Test Forecast", ax = ax)
plt.title("February 2020 Forecast vs Actuals")
plt.show()


# Model performance on X_test
mape = mean_absolute_percentage_error(X_test["y"], X_test_forecast_holidays["yhat"])
print("MAPE:%s" % round(mape, 4))


# # 4.HyperParmeter Tuning using ParameterGrid


# parameters
params_grid = {
    "seasonality_mode": ("multiplicative", "additive"),
    "changepoint_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
    "holidays_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_changepoints": [100, 150, 200],
}
grid = ParameterGrid(params_grid)
cnt = np.sum([1 for p in grid])
print("Total Possible Models: %s" % cnt)


# ## 4.1 Prophet Model Tuning


start = "2019-12-31"
end = "2020-02-26"

model_parameters = pd.DataFrame(columns = ["MAPE", "Parameters"])
for param in grid:
    print(param)
    random.seed(0)
    test = pd.DataFrame()
    train_model = Prophet(
        changepoint_prior_scale = param["changepoint_prior_scale"],
        holidays_prior_scale = param["holidays_prior_scale"],
        n_changepoints = param["n_changepoints"],
        seasonality_mode = param["seasonality_mode"],
        weekly_seasonality = True,
        daily_seasonality = True,
        yearly_seasonality = True,
        holidays = holiday,
        interval_width = 0.95
    )
    train_model.add_country_holidays(country_name = "US")
    train_model.fit(X_test)
    train_forecast = train_model.make_future_dataframe(periods = 57, freq = "D", include_history = False)
    train_forecast = train_model.predict(train_forecast)
    test = train_forecast[["ds", "yhat"]]
    Actual = df[(df["ds"] > start) & (df["ds"] <= end)]
    MAPE = mean_absolute_percentage_error(Actual["y"], abs(test["yhat"]))
    print("Mean Absolute Percentage Error(MAPE)%s" % ("-" * 30), MAPE)
    
    model_parameters = model_parameters.append({
        "MAPE": MAPE, 
        "Parameters": param
    }, ignore_index = True)


parameters = model_parameters.sort_values(by = ["MAPE"])
parameters = parameters.reset_index(drop = True)
parameters


# ### The best model


parameters["Parameters"][0]


# ## 4.2 Train the final model


# Setup and trin model with holidays
final_model = Prophet(
    holidays = holiday,
    changepoint_prior_scale = 0.5,
    holidays_prior_scale = 0.1,
    n_changepoints = 200,
    seasonality_mode = "multiplicative",
    weekly_seasonality = True,
    daily_seasonality = True,
    yearly_seasonality = True,
    interval_width = 0.95
)
final_model.add_country_holidays(country_name = "US")
final_model.fit(X_test)
future = final_model.make_future_dataframe(periods = 122, freq = "D")
forecast = final_model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7)


forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)


# plot the components of the model
fig = final_model.plot_components(forecast)


# plot the forecast
fig, ax = plt.subplots(1, figsize = (15, 5))
fig = final_model.plot(forecast, ax = ax)
plt.show()


X_test_final = final_model.predict(X_test)
X_test_final[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(7)


X_test_final[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)


# plot the forecast with the actuals
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_final, ax = ax)
plot = plt.suptitle("Forecast vs Actuals")
plt.show()


# Plot the forecast with the actuals of 2020
fig, ax = plt.subplots(1, figsize = (15, 5))
ax.scatter(X_test["ds"], X_test["y"], color = "red")
fig = model.plot(X_test_final, ax = ax)
ax.set_xbound(lower = "2020-01-01", upper = "2020-02-26")
ax.set_ylim(0, 60000)
plot = plt.suptitle("First Week of January Forecast vs Actuals")
plt.show()


# Compare the Sales and forecasted Sales
fig, ax = plt.subplots(1, figsize = (15, 5))
X_test.plot(kind = "line", x = "ds", y = "y", color = "red", label = "Test", ax = ax)
X_test_final.plot(kind = "line", x = "ds", y = "yhat", color = "green", label = "Test Forecast", ax = ax)
plt.title("February 2020 Forecast vs Actuals")
plt.show()


# Model performance on X_test
mape = mean_absolute_percentage_error(X_test["y"], X_test_final["yhat"])
print("MAPE:%s" % round(mape, 4))





