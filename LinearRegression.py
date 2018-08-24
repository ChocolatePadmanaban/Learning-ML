import pandas as pd

##pandas is a Python package providing fast, flexible, and expressive data
##structures designed to make working with “relational” or “labeled” data both
##easy and intuitive. It aims to be the fundamental high-level building block
##for doing practical, real world data analysis in Python

import quandl, math, datetime

##Get Financial Data Directly into Python
##Get millions of financial and economic
##datasets from hundreds of publishers directly into Python

import numpy as np #like pandas but for homogenius data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm # sklearn is ML FRAMEWORK 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
##Data Frames are grenerall initiated with df
##Two-dimensional size-mutable, potentially heterogeneous tabular data
##structure with labeled axes (rows and columns). Arithmetic operations
##align on both row and column labels. Can be thought of as a dict-like
##container for Series objects. The primary pandas data structure.


##The opening price is the price at which a security first trades upon the
##opening of an exchange on a given trading day; for example, the New York
##Stock Exchange opens at precisely 9:30 am Eastern time.

##A 52-week high/low is the highest and lowest price that a stock has traded at
##during the previous year. Many traders and investors view the 52-week high or
##low as an important factor in determining a stock's current value and
##predicting future price movement.

##"Closing price" generally refers to the last price at which a stock trades
##during a regular trading session. For many U.S. markets, regular trading
##sessions run from 9:30 a.m. to 4:00 p.m. Eastern Time.

##Volume is simply the number of shares or contracts that trade over a given
##period ... on to charts, which help identify trading opportunities in price movements.

##When a company declares a dividend, it sets a record date when you must be on the
##company's books as a shareholder to receive the dividend. ... The ex-dividend date
##is normally two business days before the record date. If you purchase a stock on or
##after its ex-dividend date, you will not receive the next dividend payment.

##A 3-for-1 stock split means that for every one share held by an investor, there will
##now be three. In other words, the number of outstanding shares in the market will
##triple. On the other hand, the price per share after the 3-for-1 stock split will
##be reduced by dividing the price by 3.

##An adjusted closing price is a stock's closing price on any given day of trading that
##has been amended to include any distributions and corporate actions that occurred at
##any time before the next day's open.

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1))

X = preprocessing.scale(X)
X=X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##clf = LinearRegression(n_jobs=-1)
##clf.fit(X_train, y_train)
##with open('linearregression.pickle','wb')as f:
##    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)




    
confidence = clf.score(X_test, y_test)

print(confidence)

forecast_set = clf.predict(X_lately)

print(forecast_set,confidence, forecast_out)

df['Forecast']= np.nan

last_date = df.iloc[-1].name
last_unix= last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()















