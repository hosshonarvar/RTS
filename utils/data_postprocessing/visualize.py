#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:27:53 2019

@author: hosseinhonarvar
"""

from utils.data_preparation import data_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

def plot(symbol, best_model, min_price ,max_price):
    seq_obj = data_split.MultiSequence(symbol, best_model.window_size,1)
    test_predict = best_model.model.predict(seq_obj.X)
    scaler = MinMaxScaler(feature_range=(min_price ,max_price))
    orig_data = seq_obj.original_data.reshape(-1,1)
    orig_prices = scaler.fit_transform(orig_data).flatten()
    plt.plot(orig_prices, color='k')
    length = len(seq_obj.X) + best_model.window_size 
    test_in = np.arange(best_model.window_size,length,1)
    pred_prices = scaler.fit_transform(test_predict.reshape(-1,1)).flatten()
    plt.plot(test_in,pred_prices,color = 'b')
    plt.xlabel('day')
    plt.ylabel('Closing price of stock')
    plt.title("Price prediction for {}".format(symbol))
    plt.legend(['Actual','Prediction'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()