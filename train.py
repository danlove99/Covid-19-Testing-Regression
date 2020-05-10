# univariate stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np

df = pd.read_csv('canada-testing.csv')
df = df.iloc[::-1]
X = df['day'].values
y = df['tests'].values

X = np.expand_dims(X, axis=1)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[1]))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1,1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
x_input = np.array([[45]])
x_input = np.expand_dims(x_input, axis=1)
yhat = model.predict(x_input, verbose=0)
print(str(y[43]) + "\n" + str(y[44]))
print("Next in sequence: " + str(yhat))
