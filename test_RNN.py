# import required packages
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
	model = keras.models.load_model('./models/20858688_RNN_model.h5')
	# 2. Load your testing data
	#reading the test data
	Test_Data = pd.read_csv("./data/test_data_RNN.csv")
	X_test=Test_Data.drop(['Target'],axis=1)
	y_test=Test_Data['Target']
	#applying scaling that have been applied on training model
	scaler = open("./models/scaler_RNN_model","rb")
	scaler = pickle.load(scaler)
	X_test=scaler.transform(X_test)
	#numpy conversion
	X_test=np.array(X_test)
	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
	y_test=np.array(y_test)
	# 3. Run prediction on the test data and output required plot and loss
	y_pred = model.predict(X_test)
	# print(y_pred)
	# loss_calculation=(np.abs(y_test-y_pred.ravel()).sum())/len(y_test)
	loss_calculation=mean_squared_error(y_test,y_pred)
	result_array=pd.DataFrame({'y_test':y_test, 'y_predicted':y_pred.ravel()})
	result_array[1:].plot.line(figsize=(15,10))
	plt.xlabel('Number of Samples')
	plt.ylabel('Opening price')
	plt.title('Comparioson')
	plt.show()
	print('The loss on test dataset is ',loss_calculation)