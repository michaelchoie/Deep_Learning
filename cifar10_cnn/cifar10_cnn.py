'''
Train a simple CNN on the CIFAR10 image dataset
'''

import numpy as np 
import matplotlib.pyplot as plt 
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Step 1: Load pre-shuffled train/test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# View images
fig = plt.figure(figsize = (20,5))
for i in range(36):
	ax = fig.add_subplot(3, 12, i+1, xticks = [], yticks = [])
	ax.imshow(np.squeeze(x_train[i]))
plt.show()

# Step 2: Normalize images to (0,1) scale
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# Step 3: One-hot encode target labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Step 4: Partition dataset into train, validation, test
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print(x_valid.shape[0], "validation samples")

# Step 5: Create and configure Augmented Image Classifier

datagen_train = ImageDataGenerator(width_shift_range = 0.1, # randomly shift images horizontally (10% of total width)
								   height_shift_range = 0.1, # randomly shift images vertically (10% of total height)
								   horizontal_flip = True) # randomly flip images horizontally
datagen_valid = ImageDataGenerator(width_shift_range = 0.1, 
								   height_shift_range = 0.1, 
								   horizontal_flip = True)

datagen_train.fit(x_train)
datagen_valid.fit(x_valid)

# take subset of training data
x_train_subset = x_train[:12]

# visualize subset of training data
fig = plt.figure(figsize=(20,2))
for i in range(0, len(x_train_subset)):
    ax = fig.add_subplot(1, 12, i+1)
    ax.imshow(x_train_subset[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

# visualize augmented images
fig = plt.figure(figsize=(20,2))
for x_batch in datagen_train.flow(x_train_subset, batch_size=12): # flow(): takes numpy data & label arrays, and generates batches of augmented/normalized data
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(x_batch[i])
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break;

# Step 6: Define Model Architecture
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', 
                        input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# Step 7: Compile model
model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

# Step 8: Train model
checkpointer = ModelCheckpoint(filepath = "model.weights.best.hdf5", verbose = 1, save_best_only = True)

data_augmentation = False
batch_size = 32
epochs = 5

if not data_augmentation:
	print("Not using data augmentation")
	hist = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_valid, y_valid), 
					 callbacks = [checkpointer], verbose = 2, shuffle = True) # shuffle so that mini-batches don't contain highly correlated examples
else:
	print("Using real-time data augmentation.")
	model.fit_generator(datagen_train.flow(x_train, y_train, batch_size = batch_size), 
										   steps_per_epoch = x_train.shape[0] // batch_size, epochs = epochs, # // means integer devision, not floating point
										   verbose = 2, callbacks = [checkpointer], validation_data = datagen_valid.flow(x_valid, y_valid, batch_size = batch_size),
										   validation_steps = x_valid.shape[0] // batch_size)

# Step 9: Load model with best performance
model.load_weights("model.weights.best.hdf5")

# Step 10: Calculate classification accuracy on test set
score = model.evaluate(x_test, y_test, verbose = 0)
print("\n", "Test accuracy:", score[1])

# Visualize predictions
y_hat = model.predict(x_test)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig = plt.figure(figsize = (20,8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)): # np.random.choice creates a random sample from a 1D array
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx]) # np.argmax returns indicies of maximum value
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color = ("green" if pred_idx == true_idx else "red"))
plt.show()
