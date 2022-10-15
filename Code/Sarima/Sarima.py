import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.fftpack import fft, fftfreq
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error

warnings.simplefilter('ignore', ConvergenceWarning)


class Sarima:
    def __init__(self, df, alpha=0.05):
        if isinstance(df, str):
          self.df = pd.read_csv(df)
        else:
          self.df = df.copy()
        self.col_names = list(df.keys())
        self.index_name = df.index.name
        self.start = None
        self.alpha = alpha
        self.season = self._find_season(self.df)
        self._prep_sarima()
        self.pdq = self.find_pdq(self.df, self.season)
        self.model = self.init_sarima()
        self.prediction = None
        self.conf_int = None

    def _adf_test(self,timeseries):
        #Perform Dickey-Fuller test:
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
           dfoutput['Critical Value (%s)'%key] = value
        return dfoutput['p-value']

    def _find_season(self,d):
        r = acf(d)[1:-1]
        res = argrelextrema(np.abs(r), np.greater)
        try:
            return argrelextrema(np.abs(r), np.greater)[0][0]
        except:
            return 0

    def _prep_sarima(self):
        #res = df.copy()
        #on suppose que df est déjà une copie
        p = self._adf_test(self.df)
        if p > self.alpha:
            self.df = self.df - self.df.shift(1)
            self.df = self.df.dropna()
            #clean(self.df) si jamais on a des NULL

        trend_diff = pm.arima.ndiffs(self.df,alpha=self.alpha)
        if self.season > 0:
            #self.df = self.df - self.df.shift(self.season)
            self.df = self.df - self.df.shift(trend_diff)
            self.df = self.df.dropna()
            #clean(df) si jamais on a des NULL

    def find_pdq(self,df,season):
        p = range(0, 2)
        d = range(0, 2)
        q = range(0, 2)
        if season < 2:
            season = 2
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], season) for x in list(itertools.product(p, d, q))]
        res = []
        for param in pdq:
          for param_seasonal in seasonal_pdq:
            try:
              mod = sm.tsa.statespace.SARIMAX(df,order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
              results = mod.fit(disp=False)
              res.append((param, param_seasonal, results.aic))
            except Exception as e:
              #print(e)
              pass
        try:
            res = min(res, key = lambda t: t[2]) #min sur results.aic
        except:
            res = [(0,0,0),(0,0,0,0),0]
        return res # (tuple[3],tuple[4],float)

    def init_sarima(self):
        mod = sm.tsa.statespace.SARIMAX(self.df,order=self.pdq[0], seasonal_order=self.pdq[1], enforce_stationarity=False,\
                                enforce_invertibility=False)
        res = mod.fit(disp=False)
        return res

    def predict(self,start,end,dynamic=False,alpha=0.05):
        self.alpha = alpha
        self.prediction = self.model.get_prediction(start=start, end=end, dynamic=dynamic)
        self.start=start
        self.conf_int = self.prediction.conf_int(alpha=self.alpha)
        return self.prediction,self.conf_int

    def mse(self,data=None,square_root=False):
        y_forecasted = self.prediction.predicted_mean
        y_forecasted = y_forecasted.to_frame(name=self.col_names[0])
        n = None
        if data is None:
            ref = self.df[self.start:]
        else:
            ref = data
            n = len(data)
        if n:
            if square_root:
                return np.sqrt(((y_forecasted[:n] - ref) ** 2)).mean()
            return ((y_forecasted[:n] - ref) ** 2).mean()
        else:
            if square_root:
                return np.sqrt(((y_forecasted - ref) ** 2)).mean()
            return ((y_forecasted - ref) ** 2).mean()

    def mape(self,data=None):
        if self.prediction == None:
            return None
        y_forecasted = self.prediction.predicted_mean
        y_forecasted = y_forecasted.to_frame(name=self.col_names[0])
        n = None
        if data is None:
            ref = self.df[self.start:]
        else:
            ref = data
            n = len(data)
        if n:
            return np.mean(np.abs((ref[self.col_names[0]] - y_forecasted[self.col_names[0]][:n])/y_forecasted[self.col_names[0]][:n]))*100
        return np.mean(np.abs((ref[self.col_names[0]] - y_forecasted[self.col_names[0]])/y_forecasted[self.col_names[0]]))*100

    def rmse(self,data=None):
        if self.prediction == None:
            return None
        y_forecasted = self.prediction.predicted_mean
        y_forecasted = y_forecasted.to_frame(name=self.col_names[0])
        n = None
        if data is None:
            ref = self.df[self.start:]
        else:
            ref = data
            n = len(data)
        if n:
            return mean_squared_error(ref[self.col_names[0]], y_forecasted[self.col_names[0]][:n], squared=False)
        return mean_squared_error(ref[self.col_names[0]], y_forecasted[self.col_names[0]], squared=False)
    
    def mae(self,data=None):
        if self.prediction == None:
            return None
        y_forecasted = self.prediction.predicted_mean
        y_forecasted = y_forecasted.to_frame(name=self.col_names[0])
        n = None
        if data is None:
            ref = self.df[self.start:]
        else:
            ref = data
            n = len(data)
        if n:
            return mae(ref[self.col_names[0]], y_forecasted[self.col_names[0]][:n])
        return mae(ref[self.col_names[0]], y_forecasted[self.col_names[0]])
