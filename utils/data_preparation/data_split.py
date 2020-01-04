from abc import ABC, abstractmethod
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SequenceBase(ABC):
    def __init__(self,symbol:str,window_size:int,target_length:int):
        try:
            self.__window_size = window_size
            self.__target_length = target_length
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            self.__data_normal = pd.read_csv(path_norm_data,index_col='index')
        except:
            print("Unexpected error for symbol {} : {}".format(symbol,sys.exc_info()[0]))
    
    @property
    def data(self):
        return self.__data_normal

    @property
    def original_data(self):
        return self.__data_normal['normal_close'].values

    @property
    def window_size(self):
        return self.__window_size

    @property
    def target_length(self):
        return self.__target_length

    @property
    @abstractmethod
    def X(self):
        pass
     
    @property
    @abstractmethod
    def y(self):
        pass

class SimpleSequence(SequenceBase):
    def __init__(self,symbol:str,window_size:int,target_length:int):
        SequenceBase.__init__(self,symbol,window_size,target_length)
        self.__sequence_data()
    
    def __sequence_data(self):
        close = self.data['normal_close'].values
        X=[]
        y=[]
        pointer = 0
        data_length = len(close)
        while (pointer+self.window_size+self.target_length)<=data_length:
            X.append(close[pointer:pointer+self.window_size])
            y.append(close[pointer+self.window_size:pointer+self.window_size+self.target_length])
            pointer+=1
        self.__X = np.asarray(X)
        self.__X = self.__X.reshape((-1,self.__X.shape[-1],1))
        self.__y = np.asarray(y)

    @property
    def X(self):
        return self.__X
     
    @property
    def y(self):
        return self.__y

class MultiSequence(SequenceBase):
    def __init__(self,symbol:str, window_size:int, target_length:int):
        SequenceBase.__init__(self,symbol, window_size, target_length)
        self.__sequence_data()

    def __sequence_data(self):
        close = self.data['normal_close'].values  
        returns = self.data['normal_returns'].values
        mfi = self.data['normal_mfi'].values
        X = []
        y = []
        pointer = 0
        data_length = len(close)
        while (pointer + self.window_size + self.target_length) <= data_length:
            x_close = close[pointer:pointer + self.window_size].reshape(-1,1)
            x_returns = returns[pointer:pointer + self.window_size].reshape(-1,1)
            x_mfi = mfi[pointer:pointer + self.window_size].reshape(-1,1)
            x_ = np.append(x_close,x_returns, axis=1)
            x_ = np.append(x_,x_mfi, axis=1)
            X.append(x_)
            y.append(close[pointer + self.window_size:pointer + self.window_size + self.target_length])
            pointer += 1
        self.__X = np.asarray(X)
        self.__y = np.asarray(y)

    @property
    def X(self):
        return self.__X
     
    @property
    def y(self): 
        return self.__y

def split_data(seq_obj:SequenceBase,split_rate=0.2):
    split = int(len(seq_obj.X) * (1-split_rate))
    X_train = seq_obj.X[:split,:]
    y_train = seq_obj.y[:split]

    X_test = seq_obj.X[split:,:]
    y_test = seq_obj.y[split:]
    return X_train,y_train,X_test,y_test

def graph_prediction(trained_model,X_train,X_test,original,window_size):
    train_predict = trained_model.predict(X_train)
    test_predict = trained_model.predict(X_test)
    plt.plot(original,color='k')
    split = len(X_train)
    split_pt = split + window_size
    train_in = np.arange(window_size,split_pt,1)
    plt.plot(train_in,train_predict,color='b')
    test_in = np.arange(split_pt,split_pt+len(test_predict),1)
    plt.plot(test_in,test_predict,color='r')

    plt.xlabel('day')
    plt.ylabel('(normalized) price of stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()