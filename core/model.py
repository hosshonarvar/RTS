import numpy as np
import pandas as pd
import sys

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop

def fixed_model(X, y, learn_rate):
    model = Sequential()
    model.add(LSTM(5, input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
def dynamic_model(X, y, learn_rate):
    model = Sequential()
    model.add(LSTM(X.shape[1], input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
def bidirectional_model(X, y, learn_rate):
    model = Sequential()
    model.add(Bidirectional(LSTM(X.shape[1], return_sequences=False), input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
def stacked_model(X, y, learn_rate):
    model = Sequential()
    model.add(LSTM(10, return_sequences=True, input_shape=(X.shape[1:])))
    model.add(LSTM(5))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
def final_model(X:np.array,y:np.array,learn_rate:float,dropout:float):
    model = Sequential()
    model.add(Bidirectional(LSTM(X.shape[1],return_sequences=False),input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dropout(dropout))
    model.add(Dense(y.shape[1],activation='tanh'))
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

# Create list of our models for use by the testing function.
models = []
models.append(("Fixed", fixed_model))
models.append(("Dynamic", dynamic_model))
models.append(("Bidirectional", bidirectional_model))
models.append(("Stacked", stacked_model))

class ModelLoader(object):
    def __init__(self, symbol: str):
        self.sub_folder = "./model/{0:}".format(symbol)
        self.model_path = "./model/{0:}/{0:}_model.json".format(symbol)
        self.weights_path = "./model/{0:}/{0:}_weights.h5".format(symbol)
        self.prop_path = "./model/{0:}/{0:}_train_props.json".format(symbol)
        try:
            if not os.path.isfile(self.model_path):
                print("No model exist for {}".format(symbol))
                return
            if not os.path.isfile(self.weights_path):
                print("No weigths file exist for {}".format(symbol))
                return
            if not os.path.isfile(self.prop_path):
                print("No training properties file exist for {}".format(symbol))
                return
            with open(self.model_path, 'r') as json_file:
                loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(self.weights_path)
                loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
                self.__model = loaded_model
            with open(self.prop_path, 'r') as prop_file:
                self.__train_prop = json.load(prop_file)
        except OSError as err:
            print("OS error for symbol {}: {}".format(symbol, err))
        except:
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))

    @staticmethod
    def root_path():
        return "./model"

    @property
    def model(self):
        return self.__model

    @property
    def ticker(self):
        return self.__train_prop["ticker"]

    @property
    def window_size(self):
        return self.__train_prop["window_size"]

    @property
    def train_prop(self):
        return self.__train_prop

    @staticmethod
    def save(symbol: str, model: Sequential, train_props: dict):
        try:
            if not os.path.isdir(ModelLoader.__sub_folder.format(symbol)):
                os.makedirs(ModelLoader.__sub_folder.format(symbol))
            model_json = model.to_json()
            with open(ModelLoader.__model_path.format(symbol), "w") as json_file:
                json_file.write(model_json)
            model.save_weights(ModelLoader.__weights_path.format(symbol))
            with open(ModelLoader.__prop_path.format(symbol), 'w') as prop_file:
                json.dump(train_props, prop_file)

        except OSError as err:
            print("OS error for symbol {}: {}".format(symbol, err))
        except:
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))