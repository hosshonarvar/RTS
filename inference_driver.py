#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:27:53 2019

@author: hosseinhonarvar
"""

from core.model_inference import Inference
from utils.data_preparation.data_preprocessing.data_creator import csv_creator

tickers = ['AAPL']
start_date = '20161101'
end_date = '20181031'
#download quotes from yahoo and save to directory

for ticker in tickers:
    print(ticker)
    csv_creator(ticker,start_date,end_date)

stock_1 = tickers[0]
print("***  Closing price prediction for {} *** ".format(stock_1))
pred = Inference(stock_1)
pred.select_model(verbose=1, tickers=tickers)
pred.plot_predictions()