#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:27:53 2019

@author: hosseinhonarvar
"""

from utils.data_preparation import data_fetcher
from utils.data_preparation import data_creator
import matplotlib.pyplot as plt
import pandas as pd
from utils.data_postprocessing.metrics_calculator import Sharpe_ratio

# Get the list of companies from ./data/dow30.csv
stocks = data_fetcher.companies()
print(stocks)

# Get the symbol of companies from /data/dow30.csv
symbols = stocks['Symbol'].values.tolist()
print(symbols)

start_date, end_date = '20161101', '20181031'    

# Download quotes from yahoo and save the normalized quotes to ./data/{company symbol}/quotes.csv
for ticker in symbols:
        
    data_creator.csv_creator(ticker, start_date, end_date)

### Volatility 
vols = [data_creator.Volatility(ticker).annual for ticker in symbols]
dataset = pd.DataFrame({'Symbol':symbols,'Volatility':vols})

# Volatility distribution
dataset.hist()
plt.show()

# Box plot to show range
dataset.boxplot()
plt.show()

# Volatility statistics
dataset.describe()

### Sharpe ratio 
SRs = [Sharpe_ratio(ticker).annual for ticker in symbols]
dataset = pd.DataFrame({'Symbol':symbols, 'SR':SRs})

# Box plot to show range
dataset.plot(kind='bar', x='Symbol', y='SR')
