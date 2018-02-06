import numpy as np
import pandas as pd
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

# Load in the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 1000)

print(x_train.shape)
print(x_test.shape)

# Examine the data
print(x_train[0])
print(y_train[0])

# One-hot encoding the input
tokenizer = Tokenizer(num_words = 1000) # turns input into vector, each element size = 1000
x_train = tokenizer.sequences_to_matrix(x_train, mode = 'binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode = 'binary')
print(x_train[0])

# One-hot encode the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# Build the model architecture
model = Sequential()
model.add(Dense(16, activation = 'relu', input_dim = 1000))
model.add(Dropout(0.2))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.summary

# Train the model
learning = model.fit(x_train, y_train, epochs = 20, batch_size = 500, verbose = 1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose = 1)
print("Accuracy: ", score[1])
