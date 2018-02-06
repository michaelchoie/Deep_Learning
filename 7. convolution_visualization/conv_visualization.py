import cv2
import scipy.misc
import matplotlib.pyplot as plt

# Step 1: Import the image

img_path = "Maya.jpg"

# Load color image
bgr_img = cv2.imread(img_path)

# Convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# Resize to smaller
small_img = scipy.misc.imresize(gray_img, 0.3)

# Normalize pixel values to range (0,1)
small_img = small_img.astype("float32")/255

# Plot image
plt.imshow(small_img, cmap = 'gray')
plt.show()

# Step 2: Specify the Filters

import numpy as np

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = [filter_1, filter_2, filter_3, filter_4]

# Visualize the filters
fig = plt.figure(figsize = (10, 5))
for i in range(4):
	ax = fig.add_subplot(1, 4, i+1, xticks = [], yticks = [])
	ax.imshow(filters[i], cmap = "gray")
	ax.set_title("Filter %s" % str(i+1))
	width, height = filters[i].shape
	for x in range(width):
		for y in range(height):
			ax.annotate(str(filters[i][x][y]), xy = (y,x),
						horizontalalignment = "center",
						verticalalignment = "center",
						color = "white" if filters[i][x][y] < 0 else "black")
plt.show()

# Step 3: Visualize the Activation Maps for each filter

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
import matplotlib.cm as cm 

# Plot image
plt.imshow(small_img, cmap = "gray")

# Define the neural network with a single convolutional layer with 1 filter
model = Sequential()
model.add(Convolution2D(1, (4,4), activation = "relu", input_shape = (small_img.shape[0], small_img.shape[1], 1))) # Number of filters, rows/columns in convolutional kernel, activation, input shape

# Apply convolutional filter and return output
def apply_filter(img, index, filter_list, ax):
	# Set weights of the filter in the convolutional layer to filter_list[i]
	model.layers[0].set_weights([np.reshape(filter_list[i], (4,4,1,1)), np.array([0])])
	# Plot the corresponding activation map
	ax.imshow(np.squeeze(model.predict(np.reshape(img, (1, img.shape[0], img.shape[1], 1)))), cmap = "gray") # np.squeeze removes single-dimensional entries in array

# Visualize all filters
fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(left = 0, right = 1.5, bottom = 0.8, top = 1, hspace = 0.05, wspace = 0.05) # tunes subplot layout
for i in range(4):
	ax = fig.add_subplot(1, 4, i+1, xticks = [], yticks = [])
	ax.imshow(filters[i], cmap = "gray")
	ax.set_title("Filter %s" % str(i+1))

# Visualize all activation maps
fig = plt.figure(figsize = (20,20))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    apply_filter(small_img, i, filters, ax)
    ax.set_title('Activation Map for Filter %s' % str(i+1))

plt.show()