import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from google.colab import files
file = files.upload()

# load dataset
df = pd.read_csv("housing_data.csv")
df.head()

# split into input (X) and output (Y) variables
y = df.pop('AboveMedianPrice')
x = df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# create model
model = Sequential()
model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
model_history=model.fit(X_train, y_train, batch_size = 10, epochs = 100)

# list all data in history
model.summary()
#total params are total number of weights and biases

# Predict the Test set results
Y_pred = model.predict(X_test)
Y_pred

# Calculate the Model Performance
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, Y_pred)

mae

"""### Model Accuracy = 1-0.30 = 0.70"""
