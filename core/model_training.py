from utils.data_preparation.data_split import data_split
from core.model import final_model

def train(ticker:str,window_sizes:list,learn_rates:list,dropouts:list, 
          epochs:list, batch_size:int,verbose=0):
    best_model = None
    lowest_test_error = 2.0
    best_training_error = 0.0
    best_learn_rate = 0.0
    best_dropout_rate = 0.0
    best_epoch = 0
    best_window_size = 0
    counter = 1
    if verbose==1:
        print("*** Best Model Selection for {} ***".format(ticker))
        print("=" * 60)

    for window_size in window_sizes:
        if verbose == 1:
            print("\nWindow size: {}".format(window_size))
            print('-' * 60)
        
        seq_obj = data_split.MultiSequence(ticker,window_size,1)
        X_train,y_train,X_test,y_test = data_split.split_data(seq_obj)

        for rate in learn_rates:
            for dropout in dropouts:
                for epoch in epochs:
                    model = final_model(X_train,y_train,rate,dropout)
                    model.fit(X_train,y_train,epochs=epoch,batch_size=batch_size, verbose=0)

                    training_error = model.evaluate(X_train,y_train,verbose=0)
                    testing_error = model.evaluate(X_test,y_test,verbose=0)

                    if verbose==1:
                        msg = " > Learn rate: {0:.4f} Dropout: {1:.2f}"
                        msg+= " Epoch: {2:} Training error: {3:.4f} Testing error: {4:.4f}"
                        msg = "{0:2}".format(str(counter))+"  "+msg.format(rate,dropout, epoch, training_error, testing_error)
                        print(msg)

                    if lowest_test_error>testing_error:
                        best_model = model
                        lowest_test_error = testing_error
                        best_learn_rate = rate
                        best_dropout_rate = dropout
                        best_epoch = epoch
                        best_training_error = training_error
                        best_window_size = window_size
                    
                    counter+=1
    if verbose in [1,2]:
        print("\nModel selection summary for {} with window size of {}:".format(ticker,best_window_size))
        print('-' * 60)
        msg = " ==> Learn rate: {0:.4f} Dropout: {1:.2f}"
        msg += " Epoch: {2:} Training error: {3:.4f} Testing error: {4:.4f}"
        msg = msg.format(best_learn_rate,best_dropout_rate, best_epoch, best_training_error, lowest_test_error)
        print(msg)
    
    best_dict = {}
    best_dict["ticker"] = ticker
    best_dict["test_error"] =  float("{0:.4f}".format(lowest_test_error) )  
    best_dict["learn_rate"] = best_learn_rate
    best_dict["dropout"] = best_dropout_rate
    best_dict["epoch"] = best_epoch
    best_dict["train_error"] =  float("{0:.4f}".format(best_training_error)  ) 
    best_dict["window_size"] = best_window_size
    return (best_model,best_dict)
