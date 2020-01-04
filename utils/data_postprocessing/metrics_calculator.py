import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import numpy as np

class Sharpe_ratio: # need to add the risk free signal (S&P 500)
    
    def __init__(self, symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data, index_col='index')
            return_signal = dataset['normal_returns']
            return_mean = return_signal.mean()
            return_std = return_signal.std()
            self.__SR = (return_mean/return_std)*math.sqrt(252)
        except:
            self.__SR = None

    @property
    def annual(self):
        return self.__SR
    
class Mean_squared_error:
    
    def __init__(self, symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data, index_col='index')
            close_signal = dataset['close']
            close_signal_predicted = dataset['close'] # need to correct this
            self.__MSR = mean_squared_error(close_signal, close_signal_predicted)
        except:
            self.__MSR = None
        
    @property
    def MSR(self):
        return self.__MSR
    
class Directional_accuracy:
    
    def __init__(self, symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data, index_col='index')
            close_signal = dataset['close']
            close_signal_predicted = dataset['close']
            close_signal_diff = close_signal[1:] - close_signal[:-1]
            close_signal_predicted_diff = close_signal_predicted[1:] - close_signal_predicted[:-1]
            self.__DA = np.mean((np.sign(close_signal_diff) == np.sign(close_signal_predicted_diff)).astype(int))
        except:
            self.__DA = None
        
    @property
    def DA(self):
        return self.__DA
    
    
    
    