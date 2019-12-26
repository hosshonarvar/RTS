from inference.model import Predictor
from preprocessing.preprocessing_driver import *

tickers = ['GOOG']
start_date = '20161101'
end_date = '20181031'
#download quotes from yahoo and save to directory
for ticker in tickers:
    Predictor.download_prep(ticker,start_date,end_date)

stock_1 = tickers[0]
print("***  Closing price prediction for {} *** ".format(stock_1))
pred = Predictor(stock_1)
pred.select_model(verbose=1)
pred.graph()