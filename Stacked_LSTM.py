#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:58:33 2020

@author: mahparsa
"""

# you can find a sourse code here https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
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

#to load a data set that has been saved as .mat we use scipy.io.loadmat
data = scipy.io.loadmat('sampleEEGdata')
#type() to undersyand the type of a variable
print( "the type of data is:.... ", type(data))

EEGdata = data["EEG"][0,0]["data"]
print( "the type of EEGdata  is:.... ", type(EEGdata ))
print( "the shape of EEGdata  is:.... ", np.shape(EEGdata))

#Out[81]: (64, 640, 99)
#we are choosing cahnnel 12 for 12 subjects


dataE=EEGdata[1,:,12]
    
np.random.seed(4)


train_size = int(len(dataE) * 0.6)
test_size = len(dataE) - train_size
train, test = dataE[0:train_size], dataE[train_size:len(dataE)]
print(train)
print(train.shape)
print(len(train), len(test))
# We need to do some explorations with the data
plot(dataE)


from sklearn.preprocessing import MinMaxScaler
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataE = scaler.fit_transform(dataE)
#plot(dataE)
# convert an array of values into a dataset matrix
#----------------------------------------to make a time series of dataE

#create-dataset creats a time series from a NumPy array

#the look_back shows the the number of previous time steps to use as input variables to predict the next time period 
#in this case defaulted to 1.


def create_dataset(dataE, look_back):
	dataX, dataY = [], []
	for i in range(len(dataE)-look_back-1):
		a = dataE[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataE[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 4
#to define the optimal values of look-back
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX)
print(trainX.shape)

#The input vector of the LSTM network must be in the form of: [samples, time steps, features].
#Using reshape, we define it 
#number of samples=trainX.shape[0], number of time steps 
#Currently, our data is in the form: [samples, features] and we are framing the problem as one time step for each sample. We can transform the prepared train and test input data into the expected structure using numpy.reshape() as follows:
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
print(trainX[1])
#array([[12.475137 , 10.774388 , -4.1156535,  2.4252448]], dtype=float32)
print(dataE[1:4])
#array([12.475137 , 10.774388 , -4.1156535], dtype=float32)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#to design and fit our LSTM network for this problem.

# visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 100 epochs and a batch size of 1 is used.

#stacked Model
model = Sequential()
model.add(LSTM(40, activation='relu', return_sequences=True,input_shape=(1, look_back)))
model.add(LSTM(40, activation='relu',return_sequences=True))
model.add(LSTM(5, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
#the number of epochs have been choosen 10 to decrease the computaional complexity. 
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


plt.figure()
plt.plot(testY)
pyplot.plot(testPredict, color='red')

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('EEG Prediction with a Stacked LSTM')
plt.grid(True)
plt.legend(('real values', 'predicted values'),  loc='upper right')
plt.show()

