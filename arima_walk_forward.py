# ARIMA Model using walk-forward validation

'''
AR: Autoregression. A model that uses the dependent relationship between an 
observation and some number of lagged observations.
I: Integrated. The use of differencing of raw observations (e.g. subtracting 
an observation from an observation at the previous time step) in order to make 
the time series stationary.
MA: Moving Average. A model that uses the dependency between an observation 
and a residual error from a moving average model applied to lagged observations.
'''

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('dataset.csv',index_col='date',parse_dates=['date'])
df_bids = df.groupby(['bids',pd.Grouper(freq='1d')])['impressions'].sum().unstack('bids').fillna(0)
#df_bids_stacked = pd.DateFrame(df_bids.stack())
df_035 = pd.concat([df_bids[(0.35)]],axis=1,keys=['bid_0.35']).unstack()


# Predict Function

# Split into train/test
X = df_035.values.astype(int)
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]

# seed history with training dataset
history = [x for x in X] #train
predictions = list()

# Walk-forward validation

for t in range(len(test)):
	model = ARIMA(history, order=(5,2,0)) #trend='ct'
	model_fit = model.fit()
	output = model_fit.forecast()#.astype(int)
	yhat = output[0].astype(int)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%.f, expected=%.f' % (yhat, obs))
    
# one-step out of sample forecast
# start_index = len(differenced)
# end_index = len(differenced)
# forecast = model_fit.predict(start=start_index, end=end_index)

# start_index = '2021-04-03'
# end_index = '2021-04-09'
# forecast = model_fit.predict(start=start_index, end=end_index)

# Invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Forecast next day
forecast = model_fit.forecast(steps=7)#[0]#.astype(int)


# Create a differenced series
def difference(df_035, interval=1):
	diff = list()
	for i in range(interval, len(df_035)):
		value = df_035[i] - df_035[i - interval]
		diff.append(value)
	return np.array(diff)

differenced = difference(X, 92)

# Invert the differenced forecast to something usable
# history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, 92)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1

# Evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecasts against actual outcomes
plt.title('ARIMA Walk-Forward Validation')
#plt.plot(forecast)
plt.plot(test)
plt.plot(predictions, color='red')
plt.xlabel('days')
plt.ylabel('impressions')
plt.show()

