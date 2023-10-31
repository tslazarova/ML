import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE

# pmdarima package time series
    
df = pd.read_csv('dataset.csv',index_col='date',parse_dates=['date'])
df_bids = df.groupby(['bids',pd.Grouper(freq='1d')])['impressions'].sum().unstack('bids').fillna(0)
df_035 = pd.concat([df_bids[(0.35)]],axis=1,keys=['bid_0.35'])

df_035_05 = pd.concat([df_bids[(0.35)],df_bids[(0.5)]],axis=1,keys=['bid_0.35','bids_0.5'])

new_df = df_035_05.iloc[:,:].reset_index()
new_df['date'] = pd.DataFrame(pd.to_numeric(new_df['date']))

X = pd.DataFrame(new_df.iloc[:,0])
y = pd.DataFrame(new_df.iloc[:,-1])


# split train test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)


# XGBRegressor
#model = XGBRegressor(objective='reg:squarederror', n_estimators=1000) # n_trees
model = XGBRegressor()
model.fit(X_train, y_train)
yhat = model.predict(X_test).astype(int)
print(yhat[0])

# evaluation of the model
# cross validation check
scores = cross_val_score(model, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

# rmse
rmse = np.sqrt(MSE(y_test, yhat))
print("RMSE : % f" %(rmse))

# Try to plot the results
# plt.plot
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, yhat, label="predicted")
plt.title('XGBRegressor Impressions Bid 0.35 (test_size=0.33)')
plt.legend()
plt.show()

