#!/usr/bin/env python
# coding: utf-8

import pandas as pd
weather = pd.read_csv("upload_folder/local_weather.csv")
import matplotlib.pyplot as plt
#weather
#print(weather.index)
weather.set_index('DATE', inplace=True)
#print(weather.loc["1960-01-01":"1960-01-31"])
weather.apply(pd.isnull).sum()/weather.shape[0] #models don't work well with null values, so this will help us prep dataset
core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
core_weather
core_weather.apply(pd.isnull).sum()/core_weather.shape[0] #calculated percentage of missing null values
core_weather["snow"].value_counts()
del core_weather["snow"]
core_weather["snow_depth"].value_counts()
del core_weather["snow_depth"]
core_weather[pd.isnull(core_weather["precip"])]
core_weather.loc["2024-02-27":"2024-04-29",:] #looks like dataset is 3 days old as of April 29 and that's why there's a null value for precip on 4/27
core_weather["precip"].value_counts()
core_weather["precip"] = core_weather["precip"].fillna(0)
core_weather[pd.isnull(core_weather["precip"])] #we filled all the missing precip data with 0
core_weather[pd.isnull(core_weather["temp_max"])]
core_weather = core_weather.drop('2024-04-27')
#just going to remove that day entirely as the data wasn't updated
core_weather[pd.isnull(core_weather["temp_max"])] #now we have no null values here  
core_weather.dtypes
core_weather.index
core_weather.index = pd.to_datetime(core_weather.index) #convert to datetimeindex for easier data manipulation
core_weather.index
core_weather.index.year #can sort by year month etc.
core_weather.apply(lambda x: (x==9999).sum()) #table 4 in doccumentation must check for missing values
# pip install matplotlib

core_weather[["temp_max","temp_min"]].plot()
# Data Looks good. No holes
core_weather[["precip"]].plot()
core_weather.groupby(core_weather.index.year).sum()["precip"]
# # Training Model
core_weather["target"] = core_weather.shift(-1)["temp_max"]
core_weather
core_weather = core_weather.iloc[:-1,:].copy()
core_weather #check to see if we removed last row with null valus
# pip install scikit-learn
from sklearn.linear_model import Ridge
reg = Ridge(alpha=.1)
predictors = ["precip", "temp_max", "temp_min"] #these are going to predict tomorrows max temp
train = core_weather.loc[:"2022-12-31"] #training set
test = core_weather.loc["2023-01-01":] #testing versus everything after that
reg.fit(train[predictors], train["target"])
predictions = reg.predict(test[predictors])
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test["target"], predictions)
# # Only about 2 degrees off from actual temperature!
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]
combined #shows actual values and then our predictions
combined.plot()
fig, ax = plt.subplots()
combined.plot(ax=ax)
fig.savefig('static/prediction_plot.png')  # Save the plot to a file
reg.coef_
# ### According to our results, precip has a negative impact on temp according to these resuults. Temp max is biggest input and temp min is smaller input
# Selecting the last row from your prepared dataset to use for prediction
last_day_features = core_weather.iloc[-1][["precip", "temp_max", "temp_min"]]
# Predicting the next day's maximum temperature
next_day_temp_max = reg.predict([last_day_features])
print(next_day_temp_max[0])
# ### In conclusion, our model can accurately predict 1 day in advanced to get the weather within 2 degrees. The way our model works is it uses the previous days temp_max, temp_min and precip in order to predict the next day. The model is trained on historical data but has access to the previous day stats. Since our last day in the dataset is April 25, we can predict the max temperature on April 26th . As shown by the previous cell results, the predicted max temperature for April 26, 2024 is about 68 degrees. If we go to an updated dataset from NOAA to find the actual data recorded for that day, we can see that our model was accurate and came within the expected 2 degrees. The actual max temperature was 66 degrees!! This is in line with our mean_absolute_error test and verifies that our model is accurately predicting the temperature within 2 degrees. 
# #### Future works: I'd like to be able to predict the weather a week in advance and possibly connect it to an api so that it automatically updates and we can continue to use our model.
