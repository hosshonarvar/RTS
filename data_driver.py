from utils.data_preprocessing.data_preparation import data_preparation as d_p
import os
import matplotlib.pyplot as plt
import pandas as pd

# Get the list of companies from ./data/dow30.csv
stocks = d_p.companies()
print(stocks)

# Get the symbol of companies from /data/dow30.csv
symbols = stocks['Symbol'].values.tolist()
print(symbols)

start_date, end_date = '20161101', '20181031'

# Download quotes from yahoo and save to ./data/{company symbol}/quotes.csv
for ticker in symbols:
    download = d_p.Downloader(ticker, start_date, end_date)
    download.save()
    

# Write quotes from yahoo and save to ./data/{company symbol}/quotes.csv
file_path = "./data/{}/quotes.csv"

for ticker in symbols:

    # Check if file exist first
    if True: #os.path.isfile(file_path.format(ticker)):
        
        F_S = d_p.Feature_Selection(ticker, pd.read_csv(file_path.format(ticker)))
        F_S.calculate_features()
        F_S.normalize_data()
        F_S.save_stock_data() # save to ./data/{company symbol}/quote_processed.csv
        F_S.save_normalized_data() # save to ./data/{company symbol}/normalized.csv
        

vols = [d_p.Volatility(ticker).annual for ticker in symbols]
dataset = pd.DataFrame({'Symbol':symbols,'Volatility':vols})

# Volatility distribution
dataset.hist()
plt.show()

# Box plot to show range
dataset.boxplot()
plt.show()

# Volatility statistics
dataset.describe()