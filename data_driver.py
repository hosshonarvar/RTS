#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:27:53 2019

@author: hosseinhonarvar
"""

from utils.data_preprocessing.data_preparation import data_fetcher as d_f
from utils.data_preprocessing.data_preparation import features_calculator as f_c
from utils.data_postprocessing import metrics_calculator as m_c
import matplotlib.pyplot as plt
import pandas as pd

# Get the list of companies from ./data/dow30.csv
stocks = d_f.companies()
print(stocks)

# Get the symbol of companies from /data/dow30.csv
symbols = stocks['Symbol'].values.tolist()
print(symbols)

start_date, end_date = '20161101', '20181031'

# Download quotes from yahoo and save to ./data/{company symbol}/quotes.csv
for ticker in symbols:
    download = d_f.Downloader(ticker, start_date, end_date)
    download.save()
    

# Write quotes from yahoo and save to ./data/{company symbol}/quotes.csv
file_path = "./data/{}/quotes.csv"

for ticker in symbols:

    # Check if file exist first
    if True: #os.path.isfile(file_path.format(ticker)):
        
        F_S = f_c.Feature_Selection(ticker, pd.read_csv(file_path.format(ticker)))
        F_S.calculate_features()
        F_S.normalize_data()
        F_S.save_stock_data() # save to ./data/{company symbol}/quote_processed.csv
        F_S.save_normalized_data() # save to ./data/{company symbol}/normalized.csv
        

vols = [m_c.Volatility(ticker).annual for ticker in symbols]
dataset = pd.DataFrame({'Symbol':symbols,'Volatility':vols})

# Volatility distribution
dataset.hist()
plt.show()

# Box plot to show range
dataset.boxplot()
plt.show()

# Volatility statistics
dataset.describe()