#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Class that creates a Keras CNN and saves/loads model
'''

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

def FromScratchModel(train_tensors, valid_tensors, test_tensors):
	'''
	Create a CNN from scratch
	Args:
		Partitioned tensors
	Return:
		Test tensors and fitted model
	'''
	model = Sequential()
	model.add(Conv2D(filters = 16, kernel_size = 2, padding = "same", activation = "elu", 
					 input_shape = (224, 224, 3)))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(BatchNormalization(axis = 1))
	model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "elu"))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(BatchNormalization(axis = 1))
	model.add(Conv2D(filters = 64, kernel_size = 2, padding = "same", activation = "elu"))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(BatchNormalization(axis = 1))
	model.add(GlobalAveragePooling2D())
	model.add(BatchNormalization(axis = 1))
	model.add(Dense(133, activation = "softmax"))

	model.summary()	
	'''
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
	                               verbose=1, save_best_only=True)
	
	model.fit(train_tensors, train_targets, 
	          validation_data=(valid_tensors, valid_targets),
	          epochs=5, batch_size=20, callbacks=[checkpointer], verbose=1)
	'''
	model.load_weights('saved_models/weights.best.from_scratch.hdf5')

	return test_tensors, model
	
def TransferLearningModel():
	'''
	CNN based on trained ResNet50 model
	Return:
		Test tensors and fitted model
	'''
	bottleneck_features = np.load("bottleneck_features/DogResnet50Data.npz")
	train_resnet50 = bottleneck_features["train"]
	valid_resnet50 = bottleneck_features["valid"]
	test_resnet50 = bottleneck_features["test"]

	resnet50_model = Sequential()
	resnet50_model.add(GlobalAveragePooling2D(input_shape = train_resnet50.shape[1:]))
	resnet50_model.add(BatchNormalization(axis = 1))
	resnet50_model.add(Dense(133, activation = "softmax"))
	resnet50_model.summary()

	resnet50_model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
	'''
	checkpointer = ModelCheckpoint(filepath = "saved_models/weights.best.resnet50.hdf5",
                               verbose = 1, save_best_only = True)

	resnet50_model.fit(train_resnet50, train_targets, 
	                   validation_data = (valid_resnet50, valid_targets),
	                   epochs = 20, batch_size = 20, callbacks = [checkpointer], verbose = 1)
	'''
	resnet50_model.load_weights("saved_models/weights.best.resnet50.hdf5")

	return test_resnet50, resnet50_model
