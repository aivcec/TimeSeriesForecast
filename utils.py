import pandas as pd
from pandas import DataFrame
from pandas import Series
from datetime import datetime
from pylab import figure
from pandas import read_csv
from pandas import to_datetime

from pandas import Series
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from math import sqrt

import pickle

def read_target_df(path):
    df = read_csv(path, sep=';')
    df.drop(df.columns[2], axis = 1, inplace=True)
    df.columns = ['date', 'target']
    df.target = df.target.replace(',','.', regex=True).astype(float)
    df.date = to_datetime(df.date, format='%d.%m.%Y')
    
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def scale(train, test, lbound=-1, ubound=1):
    
    # fit scaler
    scaler = MinMaxScaler(feature_range=(lbound, ubound))
    scaler = scaler.fit(train)
    
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def scale_v(train, val, test, lbound=-1, ubound=1):
    
    # fit scaler
    scaler = MinMaxScaler(feature_range=(lbound, ubound))
    scaler = scaler.fit(train)
    
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    # validate train
    val = val.reshape(val.shape[0], val.shape[1])
    val_scaled = scaler.transform(val)
    
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, val_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def print_error_info(errors):
    results = DataFrame()
    results["error"] = errors
    print(results.describe())
    results.boxplot()
    plt.show()
    print(errors)

def print_df_info(df, verbose=False):
    if verbose:
        print(df.head())
        print(df.tail())
    print(df.shape)
    print(df.dtypes)

from pandas import concat

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[df.columns[0]].shift(-i))
        if i == 0:
            names.append('var1(t)')
        else:
            names.append('var1(t+%d)' % (i))
            
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
def report_performance(test, predictions):
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)

    mae = mean_absolute_error(test, predictions)
    print('MAE: %.3f' % mae)

    fig = plt.figure(figsize=(25,10))
    plt.plot(test[:31], label='real')
    plt.plot(predictions[:31], label='predicted')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure(figsize=(25,10))
    plt.plot(test, label='real')
    plt.plot(predictions, label='predicted')
    plt.legend(loc='upper right')
    plt.show()

    return rmse

def plot_multivar_results(series, predictions, n_forecast_days, x_start=0, x_end=360):
    plt.figure(figsize=(25,10))
    
    plt.plot(series, color="b")

    for i in range(len(predictions)):
        x_val = [i - index for index in range(n_forecast_days - 1, -1, -1)]
        plt.plot(x_val, predictions[i], color='r')

    plt.xlim([x_start, x_end])
    plt.show()

def plot_multi_dataframe(df, startIndex, endIndex):
    plt.figure(figsize=(25,10))
    column_nr = df.shape[1]
    values = df.values
    for i in range(column_nr):
        plt.subplot(column_nr, 1, i+1)
        plt.plot(values[startIndex: endIndex, i])
        plt.title(df.columns[i], y=0.5, loc='right')
    
    plt.show()

def exhaustive_report(train_y, train_predictions, test_y, test_predictions):
    print("TRAIN")
    print("rmse: ", sqrt(mean_squared_error(train_y, train_predictions)))
    print("mae: ", mean_absolute_error(train_y, train_predictions))
    
    print("TEST")
    mae = mean_absolute_error(test_y, test_predictions)
    rmse =  sqrt(mean_squared_error(test_y, test_predictions))
    print("rmse: ", rmse)
    print("mae: ", mae)
    
    plt.figure(figsize=(25,10))
    plt.plot(test_y[:31], label='real')
    plt.plot(test_predictions[:31], label='predicted')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure(figsize=(25,10))
    plt.plot(test_y, label='real')
    plt.plot(test_predictions, label='predicted')
    plt.legend(loc='upper right')
    plt.show()
    
    return mae, rmse

def save_to_file(x, filename):
    output = open(filename + '.pkl', 'wb')
    pickle.dump(x, output)
    output.close()
    
def read_from_file(filename):
    file = open(filename +'.pkl', 'rb')
    x = pickle.load(file)
    file.close()
    
    return x