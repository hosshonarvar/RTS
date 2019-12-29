from core.model_inference.model_inference import Inference

tickers = ['MMM']
start_date = '20161101'
end_date = '20181031'
#download quotes from yahoo and save to directory

for ticker in tickers:
    print(ticker)
    Inference.download_prep(ticker,start_date,end_date)

stock_1 = tickers[0]
print("***  Closing price prediction for {} *** ".format(stock_1))
pred = Inference(stock_1)
pred.select_model(verbose=1, tickers=tickers)
pred.graph()