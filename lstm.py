from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#for reporting errors
import matplotlib.pyplot as plt
from pylab import figure
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils import *

def lstm_model(neurons, batch_input_shape, stateful=True, weights=None):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=batch_input_shape, stateful=stateful))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    if weights != None:
        model.set_weights(weights)

    return model

def fit_lstm(model, train, batch_size, nb_epoch, verbose=0):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    for i in range(nb_epoch):
        if verbose != 0:
            print("epoch number: {}".format(i))
        
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=verbose, shuffle=False)
        model.reset_states()

def forecast_lstm(model, X, batch_size=1):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

def experiment(df, neurons, batch_size, epoch_nr):
    df["target(t+1))"] = df.target.shift(-1)
    df.dropna(inplace=True)
    
    input_features = df.values.shape[1] - 1
    
    n_train_days = 5*365
    n_test_days = 365

    values = df.values
    train, test = values[:n_train_days, :], values[-n_test_days:, :]
    scaler, train_scaled, test_scaled = scale(train, test, -1, 1)
    
    print(train[:5, :])
    
    print(batch_size, 1, input_features)
    model = lstm_model(neurons, (batch_size, 1, input_features))
    fit_lstm(model, train_scaled, batch_size, epoch_nr, verbose=0)
    
    test_model = lstm_model(neurons, (1, 1, input_features), weights=model.get_weights())
    train_reshaped = train_scaled[:, :-1].reshape(len(train_scaled), 1, input_features)
    test_model.predict(train_reshaped, batch_size=1)
    
    # walk forward prediction
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(test_model, X)
        yhat = invert_scale(scaler, X, yhat)
        predictions.append(yhat)
        expected = test[i, -1]
    #print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        
    rmse = report_performance(test[:, -1], predictions)
    return rmse
