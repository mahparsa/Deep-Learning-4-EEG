#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:34:12 2020

@author: mahparsa

# you can find a nice code for time series prediction from here https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
"""
import numpy as np, cmath, scipy as sp
import scipy.io
import sklearn
import pandas as pd 
import matplotlib.pyplot as plt
import pandas
import math
from matplotlib.pyplot import *
from matplotlib import pyplot
#import basic functions from numpy that we'll need
from numpy import pi, sin, cos, exp, sqrt, log, log10, random, angle, real, imag, zeros, ceil, floor, absolute, linspace  
from numpy.fft import fft, ifft
from scipy import signal as sig
from scipy.signal import hilbert
#from matplotlib.pyplot import *
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import Series
from pandas import DataFrame
from pandas import concat


data = scipy.io.loadmat('sampleEEGdata')
#type() to undersyand the type of a variable
print( "the type of data is:.... ", type(data))

EEGdata = data["EEG"][0,0]["data"]
print( "the type of EEGdata  is:.... ", type(EEGdata ))
print( "the shape of EEGdata  is:.... ", np.shape(EEGdata))


dataE=EEGdata[1,:,12]
    
np.random.seed(4)

plot_acf(dataE)
pyplot.show()
plot_pacf(dataE, lags=10)
pyplot.show()
plot_pacf(dataE, lags=20)
pyplot.show()
plot_pacf(dataE, lags=200)
pyplot.show()
#np.random.seed(4)

lag=2
Step=75
temps = DataFrame(dataE)
dataframe = concat([ temps.shift((2)*Step), temps.shift(Step), temps], axis=1)
 
dataframe.columns = [ 't-150','t-75', 't']
print(dataframe)
data=DataFrame.as_matrix(dataframe)

train_size = int(len(data) * 0.6)
test_size = len(data) - train_size
train, test = data[1:train_size], data[train_size:len(data)]

print(train)
print(train.shape)
print(len(train), len(test))
# We need to do some explorations with the data

from sklearn.preprocessing import MinMaxScaler


#The input vector of the LSTM network must be in the form of: [samples, time steps, features].
#Using reshape, we define it 
#number of samples=trainX.shape[0], number of time steps 
#Currently, our data is in the form: [samples, features] and we are framing the problem as one time step for each sample. We can transform the prepared train and test input data into the expected structure using numpy.reshape() as follows:
look_back=2
trainX=train[(2)*Step-1:len(train),0:2]
trainX.shape
trainY=train[(2)*Step-1:len(train),2]
trainY.shape
plot_pacf(trainY, lags=100)
pyplot.show()
testX=test[(2)*Step-1:len(test),0:2]
testY=test[(2)*Step-1:len(test),2]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
print(trainX[1])
print(trainY[1])

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#to design and fit our LSTM network for this problem.

# visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 100 epochs and a batch size of 1 is used.

#stacked Model
model = Sequential()
model.add(Bidirectional(LSTM(28 , activation='relu'), input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2)
#the number of epochs have been choosen 10 to decrease the computaional complexity. 
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
trainScore_mse = mean_squared_error(trainY, trainPredict)
print('Train Score: %.2f RMSE' % (trainScore))
print('Train Score: %.2f MSE' % (trainScore_mse))

testScore = math.sqrt(mean_squared_error(testY, testPredict))
testScore_mse = mean_squared_error(testY, testPredict)

print('Test Score: %.2f RMSE' % (testScore))
print('Test Score: %.2f MSE' % (testScore_mse))


mae = mean_absolute_error(testY, testPredict)
print('MAE: %f' % mae)


expected = testY
predictions = testPredict
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
mean_forecast_error = sum(forecast_errors)/float(len(forecast_errors))
print('Forecast Errors: %s' % mean_forecast_error)
bias = sum(forecast_errors) * 1.0/len(expected)
print('Bias: %f' % bias)



plt.figure()
plt.plot(testY)
pyplot.plot(testPredict, '--',color='red')

plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.title('EEG Prediction with a Stacked LSTM')
plt.grid(True)
plt.legend(('real values', 'predicted values'),  loc='upper right')
plt.show()

plt.figure()
plt.plot(trainY)
pyplot.plot(trainPredict,'.', color='green')
plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.title('EEG Prediction with a Stacked LSTM')
plt.grid(True)
plt.legend(('real values', 'predicted values'),  loc='upper right')
plt.show()