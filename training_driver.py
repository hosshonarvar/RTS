#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:27:53 2019

@author: hosseinhonarvar
"""

from core.model_training.model_training import train
from core.models.model import ModelLoader
from utils.data_preprocessing.data_preparation import data_fetcher as d_f


stocks = d_f.companies()
symbols = stocks['Symbol'].values.tolist()
print(symbols)

window_sizes = [5,7,10]
dropouts =  [0.25,0.4]
learn_rates = [0.01,0.001]
epochs = [100,200]
batch_size = 50

result = train(symbols[0], window_sizes, learn_rates, dropouts, epochs, batch_size,verbose=1)

print("\nResults : ")
print("-"*60)
print(result[0])
print(result[1])

ModelLoader.save(result[1]['ticker'],result[0],result[1])
print("Saved trained model for {}".format(result[1]['ticker']))
