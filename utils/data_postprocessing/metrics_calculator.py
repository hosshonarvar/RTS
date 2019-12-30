import math
import pandas as pd

class Volatility(object):
    def __init__(self,symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data,index_col='index')
            self.__volatility = dataset['returns'].std() * math.sqrt(252)
        except:
            self.__volatility = -1

    @property
    def annual(self):
        return self.__volatility