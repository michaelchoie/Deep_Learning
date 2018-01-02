import pickle
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_data import load_pickle_file

def build_neural_net(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
	'''
	Create a neural network and train/test/visualize it 
	'''

	# Set layers' node count
	features_count = 784 # All the pixels in the image (28 * 28 = 784)
	hidden_count = 50
	labels_count = 10

	# Set the features and labels tensors
	features = tf.placeholder(tf.float32)
	labels = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)

	# Set the weights and biases tensors
	weights = [tf.Variable(tf.truncated_normal((features_count, hidden_count))),
			   tf.Variable(tf.truncated_normal((hidden_count, labels_count)))]
	biases = [tf.Variable(tf.zeros(hidden_count)),
			  tf.Variable(tf.zeros(labels_count))]

	# Verify that neural network inputs are valid
	test_inputs(features, labels, weights, biases)

	# Feed dicts for training, validation, and test session
	train_feed_dict = {features: train_features, labels: train_labels, keep_prob: np.float32(0.5)}
	valid_feed_dict = {features: valid_features, labels: valid_labels, keep_prob: np.float32(1.0)}
	test_feed_dict = {features: test_features, labels: test_labels, keep_prob: np.float32(1.0)}

	# Add layers
	hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0]) 
	hidden_layer = tf.nn.relu(hidden_layer)
	hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
	logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
	prediction = tf.nn.softmax(logits)

	# Cross entropy
	cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices = 1) # reduction_indicies = 1 sums tensors by row

	# Training loss
	loss = tf.reduce_mean(cross_entropy)

	# Determine if the predictions are correct
	is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

	# Calculate the accuracy of the predictions
	accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

	# Change if you have memory restrictions
	batch_size = 128

	# Find the best parameters for each configuration
	epochs = 4
	learning_rate = 0.2

	# Gradient Descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

	# The accuracy measured against the validation set
	validation_accuracy = 0.0

	# Measurements use for graphing loss and accuracy
	log_batch_step = 50
	batches = []
	loss_batch = []
	train_acc_batch = []
	valid_acc_batch = []

	# Create an operation that initializes all variables
	init = tf.global_variables_initializer()

	with tf.Session() as session:
	    session.run(init)
	    batch_count = int(math.ceil(len(train_features)/batch_size))

	    for epoch_i in range(epochs):
	        
	        # Progress bar
	        batches_pbar = tqdm(range(batch_count), desc = 'Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit = 'batches')
	        
	        # The training cycle
	        for batch_i in batches_pbar:
	            # Get a batch of training features and labels
	            batch_start = batch_i*batch_size
	            batch_features = train_features[batch_start:batch_start + batch_size]
	            batch_labels = train_labels[batch_start:batch_start + batch_size]

	            # Run optimizer and get loss
	            _, l = session.run(
	                [optimizer, loss],
	                feed_dict={features: batch_features, labels: batch_labels, keep_prob: np.float32(0.5)})

	            # Log every 50 batches
	            if not batch_i % log_batch_step:
	                # Calculate Training and Validation accuracy
	                training_accuracy = session.run(accuracy, feed_dict = train_feed_dict)
	                validation_accuracy = session.run(accuracy, feed_dict = valid_feed_dict)

	                # Log batches
	                previous_batch = batches[-1] if batches else 0
	                batches.append(log_batch_step + previous_batch)
	                loss_batch.append(l)
	                train_acc_batch.append(training_accuracy)
	                valid_acc_batch.append(validation_accuracy)

	        # Check accuracy against Validation data
	        validation_accuracy = session.run(accuracy, feed_dict = valid_feed_dict)

	# Visualize performance on validation set
	graph_loss_accuracy(batches, loss_batch, train_acc_batch, valid_acc_batch, validation_accuracy)

	# Pass test set on neural network
	test_accuracy = 0.0

	with tf.Session() as session:
	
	    session.run(init)
	    batch_count = int(math.ceil(len(train_features)/batch_size))

	    for epoch_i in range(epochs):
	        
	        # Progress bar
	        batches_pbar = tqdm(range(batch_count), desc = 'Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit = 'batches')
	        
	        # The training cycle
	        for batch_i in batches_pbar:
	            # Get a batch of training features and labels
	            batch_start = batch_i*batch_size
	            batch_features = train_features[batch_start:batch_start + batch_size]
	            batch_labels = train_labels[batch_start:batch_start + batch_size]

	            # Run optimizer
	            _ = session.run(optimizer, feed_dict = {features: batch_features, labels: batch_labels, keep_prob: np.float32(1.0)})

	        # Check accuracy against Test data
	        test_accuracy = session.run(accuracy, feed_dict = test_feed_dict)

	# assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
	print('Test Accuracy is {}'.format(test_accuracy))

def graph_loss_accuracy(batches, loss_batch, train_acc_batch, valid_acc_batch, validation_accuracy):
	'''
	Visualize neural network performance in terms of accuracy
	'''

	loss_plot = plt.subplot(211)
	loss_plot.set_title('Loss')
	loss_plot.plot(batches, loss_batch, 'g')
	loss_plot.set_xlim([batches[0], batches[-1]])
	acc_plot = plt.subplot(212)
	acc_plot.set_title('Accuracy')
	acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
	acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
	acc_plot.set_ylim([0, 1.0])
	acc_plot.set_xlim([batches[0], batches[-1]])
	acc_plot.legend(loc=4)
	plt.tight_layout()
	plt.show()

	print('Validation accuracy at {}'.format(validation_accuracy))

def test_inputs(features, labels, weights, biases):
	'''
	Confirm that inputs in neural network are correct
	'''

	from tensorflow.python.ops.variables import Variable
	
	# Test Cases
	assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
	assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
	for i in range(len(weights)):
		assert isinstance(weights[i], Variable), 'weights must be a TensorFlow variable'
		assert isinstance(biases[i], Variable), 'biases must be a TensorFlow variable'	
	assert features._shape == None or \
		   (features._shape.dims[0].value is None and features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
	assert labels._shape  == None or \
		   (labels._shape.dims[0].value is None and labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
	assert weights[0]._variable._shape == (784, 50), 'The shape of weights is incorrect'
	assert biases[1]._variable._shape == (10), 'The shape of biases is incorrect'
	assert features._dtype == tf.float32, 'features must be type float32'
	assert labels._dtype == tf.float32, 'labels must be type float32'

if __name__ == '__main__':
	train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_pickle_file('notMNIST.pickle')
	build_neural_net(train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
