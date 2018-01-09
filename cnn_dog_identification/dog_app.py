#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Process images and classify humans and dogs
'''

import os
import random
import numpy as np 
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob # finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
from PIL import ImageFile # provides support functions for the image open and save functions
from Detector import HumanDetector, DogDetector
from Detector import path_to_tensor, paths_to_tensor
from cnn import FromScratchModel, TransferLearningModel

def load_dataset(img_path):
	data = load_files(img_path)
	dog_files = np.array(data["filenames"])
	dog_targets = np_utils.to_categorical(np.array(data["target"]), 133)
	return dog_files, dog_targets

if __name__ == "__main__":

	os.chdir("/Users/MichaelChoie/Desktop/Data Science/deep-learning/dog-project")
	random.seed(8675309)

	# Step 0: Import dataset
	train_files, train_targets = load_dataset("dogImages/train")
	valid_files, valid_targets = load_dataset("dogImages/valid")
	test_files, test_targets = load_dataset("dogImages/test")

	dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))] # Removes path string

	print("There are %d total dog categories." % len(dog_names))
	print("There are %d total dog images." % len(np.hstack([train_files, valid_files, test_files])))
	print("There are %d training dog images." % len(train_files))
	print("There are %d validation dog images." % len(valid_files))
	print("There are %d test dog images."% len(test_files))

	human_files = np.array(glob("lfw/*/*"))
	random.shuffle(human_files) # shuffle to mitigate chance of highly correlated mini-batches

	print("There are %d total human images." % len(human_files))
	
	human_files_short = human_files[:100]
	dog_files_short = train_files[:100]
	
	# Detect humans
	human_classifier = HumanDetector()
	human_classifier.display_face(human_files[3])

	# Detect dogs
	dog_classifier = DogDetector()
	print("Out of the images, %.2f%% are human." % human_classifier.count_images(human_files_short))
	print("Out of the images, %.2f%% are dogs." % dog_classifier.count_images(dog_files_short))
	
	# ImageFile.LOAD_TRUNCATED_IMAGES = True         

	# Pre-process the data for Keras and create from scratch model
	train_tensors = paths_to_tensor(train_files).astype('float32') / 255
	valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
	test_tensors = paths_to_tensor(test_files).astype('float32') / 255

	test_tensors, model = FromScratchModel(train_tensors, valid_tensors, test_tensors)
	dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
	test_accuracy = 100 * np.sum(np.array(dog_breed_predictions) == np.argmax(test_targets, axis = 1)) / len(dog_breed_predictions)
	print('Test accuracy: %.2f%%' % test_accuracy)

	# Transfer Learning Model
	test_resnet50, resnet50_model = TransferLearningModel()
	resnet50_predictions = [np.argmax(resnet50_model.predict(np.expand_dims(feature, axis = 0))) for feature in test_resnet50]
	test_accuracy = 100 * np.sum(np.array(resnet50_predictions) == np.argmax(test_targets, axis = 1)) / len(resnet50_predictions)
	print('Test accuracy: %.2f%%' % test_accuracy)

	