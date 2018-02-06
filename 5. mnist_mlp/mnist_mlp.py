from keras.datasets import mnist

# Step 1: Load MNIST data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("The MNIST database has a training set of %d examples" % len(X_train))
print("The MNIST database has a testing set of %d examples" % len(X_test))

# Step 2: Visualize the data (first 6 images)

import matplotlib.pyplot as plt 						
import matplotlib.cm as cm # Colormap
import numpy as np

fig = plt.figure(figsize = (20, 20)) # Create figure instance with width, height specified
for i in range(6):
	ax = fig.add_subplot(1, 6, i+1, xticks = [], yticks = []) # 1x6 grid, i+1 subplot
	ax.imshow(X_train[i], cmap = "gray") # display image on the axes (cmap = colormap)
	ax.set_title(str(y_train[i]))

# Step 3: Visualize more detailed image

def visualize_input(img, ax):
	ax.imshow(img, cmap = "gray")
	width, height = img.shape
	thresh = img.max()/2.5
	for x in range(width):
		for y in range(height):
			ax.annotate(str(round(img[x][y], 2)), xy = (y,x), # text to annotate, sequence specifying which point to annotate
						horizontalalignment = "center",
						verticalalignment = "center",
						color = "white" if img[x][y] < thresh else "black")

fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111)
visualize_input(X_train[0], ax)
plt.show() # necessary if you want to see plot in non-interactive mode i.e terminal

# Step 4: Rescale images to (0,1)

X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

# Step 5: Encode categorical integer labels via one-hot-encoding

from keras.utils import np_utils # numpy related utilities

print("Integer valued labels: \n", y_train[:10])

y_train = np_utils.to_categorical(y_train, 10) # Converts a class vector (integers) to binary class matrix with 10 classes
y_test = np_utils.to_categorical(y_test, 10)

print("One-hot labels: \n", y_train[:10])

# Step 6: Define model architecture

from keras.models import Sequential # Linear stack of layers
from keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Flatten(input_shape = X_train.shape[1:])) # Decompose matrix into an array
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))

model.summary()

# Step 7: Compile the model

model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

'''
RMSprop is an gradient descent optimizer algorithm that resolves Adagrad's diminishing learning rate problem.
Adagrad adapts learning rate to parameters, performing larger updates to infrequent and smaller updates to frequent parameters.
This way, it handles sparse data well. 
Adagrad's problem is that its formula contains sum of squared gradients in the denominator, which makes the learning rate infinitesimal
RMSprop uses an exponentially decaying average rather than sum of squared gradients
'''

# Step 8: Calculate the classification

score = model.evaluate(X_test, y_test, verbose = 0) # verbose = 0: silent; 1: progress bar; 2: 1 line/epoch
accuracy = 100 * score[1]

# Step 9: Train the model

from keras.callbacks import ModelCheckpoint 

'''
Because neural nets take so long to train, we checkpoint our models to make sure not everything is lost in case of system failure.
We can specify the name of the file/filepath to save the model weights and when to do so. 

hdf5 = hierarchical model format, designed to store and organize large amounts of data
'''

checkpointer = ModelCheckpoint(filepath = "mnist.model.best.hdf5", verbose = 1, save_best_only = True)
hist = model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.2, callbacks = [checkpointer],
				 verbose = 1, shuffle = True)

# Step 10: Load model with best classification accuracy on test set

model.load_weights("mnist.model.best.hdf5")

# Step 11: Calculate classification accuracy on the test set

score = model.evaluate(X_test, y_test, verbose = 0)
accuracy = 100 * score[1]

print("Test Accuracy: %.4f%%" % accuracy)

