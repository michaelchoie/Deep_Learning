import hashlib												# Check data integrity of data transmission
import os
import pickle												# Serialization of Python objects
from urllib.request import urlretrieve						# Fetch data from internet
import numpy as np
from PIL import Image 										# Represent images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm										# Create progress bar
from zipfile import ZipFile 								# Manipulate zip files

def download(url, file):
	'''
	Download file from url
	:param url: URL to file
	:param file: Local file path
	'''

	if not os.path.isfile(file):
		print('Downloading ' + file + '...')
		urlretrieve(url, file)
		print('Download finished.')
	else:
		print('%s already exists!' % file)

def uncompress_features_labels(file):
	'''
	Uncompress featuers and labels from zip file
	:param file: The zip file to extract data from
	:return: features and labels as numpy arrays
	'''

	features = []
	labels = []

	with ZipFile(file) as zipf:
		# Insert progress bar
		filenames_pbar = tqdm(zipf.namelist(), unit = 'files')

		# Get features and labels from all files
		for filename in filenames_pbar:
			# Check if file is a directory
			if not filename.endswith('/'):
				with zipf.open(filename) as image_file:
					image = Image.open(image_file)
					image.load()
					# Load image as 1D float array to save memory
					feature = np.array(image, dtype = np.float32).flatten()

				# Get the letter of the image by retrieving letter from filename
				label = os.path.split(filename)[1][0]

				features.append(feature)
				labels.append(label)

	return np.array(features), np.array(labels)

def test_download(filenames, hashvalues):
	'''
	Make sure files aren't corrupted - hash values uniquely identify data and verify if data was altered
	:param filenames: dictionary with filenames mapped to train/test keys
	:param hashvalues: dictionary with hash values mapped to train/test keys
	'''

	for i in filenames:
		assert hashlib.md5(open(filenames[i], 'rb').read()).hexdigest() == hashvalues[i], \
						   '%s is corrupted. Remove the file and try again.' % filenames[i]

	print("Downloads verified as uncorrupted!")

def retrieve_data(urls, filenames, hashvalues):
	'''
	Download training and test dataset and uncompress them
	:param urls: dictionary with URLS mapped to train/test keys
	:param filenames: dictionary with filenames mapped to train/test keys
	:return: train/test labels and features
	'''

	# Download files
	download(urls['train'], filenames['train'])
	download(urls['test'], filenames['test'])
	
	print('All files downloaded')

	test_download(filenames, hashvalues)

	# Retrieve features and labels from files
	train_features, train_labels = uncompress_features_labels(filenames['train'])
	test_features, test_labels = uncompress_features_labels(filenames['test'])

	print('All features and labels have been decompressed.')

	return train_features, train_labels, test_features, test_labels

def normalize_grayscale(image_data):
    '''
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    '''

    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255

    return a + ((image_data - greyscale_min) * (b - a) / (greyscale_max - greyscale_min))

def test_normalize_grayscale():
	'''
	Make sure that the noramlize_greyscale function is working properly
	'''

	np.testing.assert_array_almost_equal(normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
										[0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     									0.125098039216, 0.128235294118, 0.13137254902, 0.9],
     									decimal = 3)
	np.testing.assert_array_almost_equal(normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
										[0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     									0.896862745098, 0.9])

	print('Grayscale normalization function passed tests!')

def fix_features_labels(is_features_normal, is_labels_encod, train_features, train_labels, test_features, test_labels):
	'''
	If features are not normalized and labels encoded, normalize and encode
	:params/return: train/test features and labels, is_features_normals/is_labels_encod turned to True
	'''

	test_normalize_grayscale()

	if not is_features_normal:
		train_features = normalize_grayscale(train_features)
		test_features = normalize_grayscale(test_features)
		is_features_normal = True 

	if not is_labels_encod:
		# Turn labels into numbers and apply one-hot-encoding
		encoder = LabelBinarizer()
		encoder.fit(train_labels) # get classes from labels 
		train_labels = encoder.transform(train_labels) # transform multi-class labels to binary 
		test_labels = encoder.transform(test_labels)

		# Change data type to float32 so we can multiply against features in TensorFlow
		train_labels = train_labels.astype(np.float32)
		test_labels = test_labels.astype(np.float32)
		is_labels_encod = True

	assert is_features_normal, 'Features not normalized'
	assert is_labels_encod, 'Labels not encoded'

	print('Features normalized and labels encoded')

	return is_features_normal, is_labels_encod, train_features, train_labels, test_features, test_labels

def partition_data(train_features, train_labels):
	'''
	Get randomized datasets for training and validation sets
	:param train_features: Independent variables
	:param train_labels: Target variable values
	:param test_size: What % of dataset to use as validation
	:param random_state: Setting seed value
	:return: Train/Validation features and labels
	'''

	train_features, valid_features, train_labels, valid_labels = train_test_split(train_features, 
																				  train_labels, 
																				  test_size = 0.05, 
																				  random_state = 832289)

	print('Training features and labels randomized and split.')

	return train_features, valid_features, train_labels, valid_labels

def pickle_file(filename):
	'''
	Save data to pickle file for easy access
	Serialized file = byte stream = 8 bits composed of binary values 
	:param filename: Name of file of serialized Python objects
	'''

	pickle_file = filename
	if not os.path.isfile(pickle_file):
		print('Saving data to pickle file...')
		try:
			with open(filename, 'wb') as pfile: # wb means write in binary
				pickle.dump(
					{
						'train_dataset': train_features,
						'train_labels': train_labels,
						'valid_dataset': valid_features,
						'valid_labels': valid_labels,
						'test_dataset': test_features,
						'test_labels': test_labels
					},
					pfile, pickle.HIGHEST_PROTOCOL) # highest protocol means most recent Python version is needed to read
		except Exception as e:
			print('Unable to save data to', pickle_file, ':', e)
			raise

	print('Data cached in pickle file')

def load_pickle_file(filename):
	'''
	Reload pickle file data 
	:param filename: Name of pickle file
	:return: Contents of pickle file (train/validation/test data)
	'''

	pickle_file = filename
	with open(pickle_file, 'rb') as f:
		pickle_data = pickle.load(f)
		train_features = pickle_data['train_dataset']
		train_labels = pickle_data['train_labels']
		valid_features = pickle_data['valid_dataset']
		valid_labels = pickle_data['valid_labels']
		test_features = pickle_data['test_dataset']
		test_labels = pickle_data['test_labels']

	print('Data loaded.')

	return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

if __name__ == '__main__':

	urls = {'train': 'https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 
			'test': 'https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip'}

	filenames = {'train': 'notMNIST_train.zip', 
				 'test': 'notMNIST_test.zip'}

	hashvalues = {'train': 'c8673b3f28f489e9cdf3a3d74e2ac8fa',
				  'test': '5d3c7e653e63471c88df796156a9dfa9'}

	# Load data
	train_features, train_labels, test_features, test_labels = retrieve_data(urls, filenames, hashvalues)

	# Limit amount of data to work with
	docker_size_limit = 150000
	train_features, train_labels = resample(train_features, train_labels, n_samples = docker_size_limit)

	# Set flags for feature engineering
	is_features_normal = False
	is_labels_encod = False

	# Data wrangling	
	is_features_normal, is_labels_encod, train_features, train_labels, test_features, test_labels = fix_features_labels(is_features_normal, is_labels_encod, train_features, train_labels, test_features, test_labels)
	train_features, valid_features, train_labels, valid_labels = partition_data(train_features, train_labels)

	# Save as pickle file
	pickle_file('notMNIST.pickle')
