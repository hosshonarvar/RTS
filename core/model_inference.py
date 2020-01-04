import pandas as pd
import os
from core.model import ModelLoader
from utils.data_preparation.data_split import MultiSequence
from utils.data_postprocessing import visualize

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

    def select_model(self,verbose=0, tickers=[]):
        tickers = [x.split('\\')[1] for x,_,_ in os.walk(ModelLoader.root_path()) if len(x.split('\\')) > 1]
        best_model = None
        lowest_test_error = 2.0
        for idx,ticker in enumerate(tickers,1):
            try:
                loaded_model = ModelLoader(ticker)
                seq_obj = MultiSequence(self.symbol,loaded_model.window_size,1)
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

    def plot_predictions(self):
        visualize.plot(self.symbol, self.__best_model, self.__min_price , 
                    self.__max_price)
