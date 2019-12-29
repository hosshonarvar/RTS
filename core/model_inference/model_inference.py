import numpy as np
import pandas as pd
import os
import sys
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Bidirectional
from keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from core.models.model import ModelLoader
from utils.data_preprocessing.data_split import data_split as d_s
from utils.data_preprocessing.data_preparation import data_preparation as d_p

class Inference(object):
    def __init__(self,symbol:str):
        self.__symbol = symbol
        file_loc = "./data/{0:}/normalized.csv".format(self.symbol)
        if os.path.isfile(file_loc):
            data = pd.read_csv(file_loc)
            self.__closing_prices = data["close"].values
            self.__max_price = max(self.__closing_prices)
            self.__min_price = min(self.__closing_prices)
        else:
            print("File does not exist : ",file_loc)
    
    @property
    def symbol(self):
        return self.__symbol
    
    @staticmethod
    def download_prep(symbol:str,start_date:str,end_date:str):
        download = d_p.Downloader(symbol,start_date,end_date)
        download.save()
        file_path = "./data/{}/quotes.csv"
        if os.path.isfile(file_path.format(symbol)):
            feature = d_p.Feature_Selection.read_csv(symbol,file_path.format(symbol))
            feature.calculate_features()
            feature.normalize_data()
            feature.save_stock_data()
            feature.save_normalized_data()
        else:
            print("File does not exist : ",file_path.format(symbol))

    def select_model(self,verbose=0, tickers=[]):
        tickers = [x.split('\\')[1] for x,_,_ in os.walk(ModelLoader.root_path()) if len(x.split('\\')) > 1]
        best_model = None
        lowest_test_error = 2.0
        for idx,ticker in enumerate(tickers,1):
            try:
                loaded_model = ModelLoader(ticker)
                seq_obj = d_s.MultiSequence(self.symbol,loaded_model.window_size,1)
                testing_error = loaded_model.model.evaluate(seq_obj.X,seq_obj.y, verbose=0)
                if verbose==1:
                    print(">{0:>3}) Now checking model: {1:<5}  Test error result: {2:.4f}".format(idx,ticker, testing_error))
                if lowest_test_error > testing_error:
                    best_model = loaded_model
                    lowest_test_error = testing_error
            except:
                pass
        self.__best_model = best_model
        self.__test_error = lowest_test_error
        if verbose in [1,2]:
            print("==> Best model ticker {0:} with error of {1:.4f}".format(self.__best_model.ticker,self.__test_error))


    def graph(self):
        seq_obj = d_s.MultiSequence(self.symbol, self.__best_model.window_size,1)
        test_predict = self.__best_model.model.predict(seq_obj.X)
        scaler = MinMaxScaler(feature_range=(self.__min_price ,self.__max_price))
        orig_data = seq_obj.original_data.reshape(-1,1)
        orig_prices = scaler.fit_transform(orig_data).flatten()
        plt.plot(orig_prices, color='k')
        length = len(seq_obj.X) + self.__best_model.window_size 
        test_in = np.arange(self.__best_model.window_size,length,1)
        pred_prices = scaler.fit_transform(test_predict.reshape(-1,1)).flatten()
        plt.plot(test_in,pred_prices,color = 'b')
        plt.xlabel('day')
        plt.ylabel('Closing price of stock')
        plt.title("Price prediction for {}".format(self.symbol))
        plt.legend(['Actual','Prediction'],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
