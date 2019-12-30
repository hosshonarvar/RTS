import sys
import os
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import numpy as np

class BaseData(object):
    def __init__(self,symbol:str):
        self.__symbol = symbol
    
    @property
    def symbol(self):
        return self.__symbol

    def save(self,file_dir:str,file_name:str,data:pd.DataFrame):
        try:
            if data is None:
                return
            full_path = os.path.join(file_dir,file_name)
            include_index = False if data.index.name == None else True
            if os.path.isdir(file_dir):
                data.to_csv(full_path,index=include_index)
            else:
                os.makedirs(file_dir)
                data.to_csv(full_path,index=include_index)
        except OSError as err:
            print("OS error for symbol {} : {}".format(self.symbol,err))
        except:
            print("Unexpected error for symbol {} : {}".format(self.symbol, sys.exc_info()[0]))

class Feature_Selection(BaseData):
    def __init__(self,symbol:str,data:pd.DataFrame,mfi_days=14):
        BaseData.__init__(self,symbol)
        self.__days = mfi_days
        self.__data = None
        self.__data_normal = None

        cols = data.columns.values
        cols_check = "Date,Open,High,Low,Close,Adj Close,Volume".split(',')
        missing = False
        for col in cols:
            found = False
            for name in cols_check:
                if col == name:
                    found = True
                    break
            if not found:
                print("The column {} is missing.".format(col))
                missing = True
                break
        if not missing:
            self.__data = data
            self.__data['Date'] = pd.to_datetime(self.__data['Date'])
            self.__data.sort_values('Date',inplace=True)
            self.__data.reset_index(drop=True,inplace=True)
            self.__data.index.name = 'index'
    
#    @staticmethod
    @classmethod
#    def read_csv(self,symbol:str,file_loc:str):
    def read_csv(cls,symbol:str,file_loc:str):
        try:
#            self.__data = pd.read_csv(file_loc)
            data = pd.read_csv(file_loc)
            return cls(symbol,data)
#            self.__symbol = symbol
        except OSError as err:
            print("OS error {}".format(err))
            return None

    @property
    def data(self):
        return self.__data
    
    @property
    def data_normal(self):
        return self.__data_normal

    def calculate_features(self):
        self.__cal_log_return("Adj Close")
        self.__cal_mfi()

    def __scale_data(self,col_Name:str):
        values = self.__data[col_Name].iloc[self.__days:].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(values).flatten()

    def __flatten_data(self,col_Name:str):
        return self.__data[col_Name].iloc[self.__days:].values.flatten()

    def normalize_data(self):
        index = self.__data.index.values[self.__days:]
        table = OrderedDict()
        table['close'] = self.__flatten_data('Adj Close')
        table['returns'] = self.__flatten_data('Adj Close_log_returns')
        table['mfi'] = self.__flatten_data('mfi_index')
        table['normal_close'] = self.__scale_data('Adj Close')
        table['normal_returns'] = self.__scale_data('Adj Close_log_returns')
        table['normal_mfi'] = self.__scale_data('mfi_index')
        self.__data_normal = pd.DataFrame(table,index=index)
        self.__data_normal.index.name = 'index'

    def __cal_log_return(self,col_name:str):
        values = self.__data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        self.__data[col_name+"_log_returns"] = pd.Series(log_returns, index = self.__data.index)

    def save_stock_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"quote_processed.csv",self.__data_normal)

    def save_normalized_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"normalized.csv",self.__data_normal)

    def __cal_mfi(self):
        typ_price = pd.DataFrame((self.__data["High"] + self.__data["Low"] + self.__data["Adj Close"])/3, columns =["price"] )
        typ_price['volume'] = self.__data["Volume"]
        typ_price['pos'] = 0
        typ_price['neg'] = 0
        typ_price['mfi_index'] = 0.0
        for idx in range(1,len(typ_price)):
            
            # Calculate the positive raw money on each day based on the price and volume
            if typ_price['price'].iloc[idx] > typ_price['price'].iloc[idx-1]:
                typ_price.at[idx,'pos' ] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]  
                
            # Calculate the negative raw money on each day based on the price and volume
            else:
                typ_price.at[idx,'neg'] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]

        pointer = 1
        for idx in range(self.__days,len(typ_price)):
            pos = typ_price['pos'].iloc[pointer:idx + 1].sum()
            neg = typ_price['neg'].iloc[pointer:idx + 1].sum()
            
            if neg != 0:
                base = (1.0 + (pos/neg))
            else:
                base = 1.0
            typ_price.at[idx,'mfi_index'] = 100.0 - (100.0/base )
            pointer += 1

        self.__data["mfi_index"] = pd.Series(typ_price["mfi_index"].values, index = typ_price.index)