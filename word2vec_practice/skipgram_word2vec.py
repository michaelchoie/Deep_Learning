"""
Build, train, and visualize word2vec network
"""

import numpy as np
import random
import os
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import helper

def build_graph(train_graph, int_to_vocab, train_words):

	# Build graph
	with train_graph.as_default():
		inputs = tf.placeholder(tf.int32, [None], name = "inputs")
		labels = tf.placeholder(tf.int32, [None, None], name = "labels")

		# Embedding layer
		n_vocab = len(int_to_vocab)
		n_embedding = 200
	
		embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
		embed = tf.nn.embedding_lookup(embedding, inputs) 

		# Negative Sampling
		n_sampled = 100
	
		softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev = 0.1))
		softmax_b = tf.Variable(tf.zeros(n_vocab))

		loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, n_vocab)
		cost = tf.reduce_mean(loss)
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# Validation
		valid_size = 16
		valid_window = 100

		valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
		valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Use cosine distance as measure of similarity
		norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims = True))
		normalized_embedding = embedding / norm 
		valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
		similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

		# Checkpoints
		saver = tf.train.Saver()

	#return train_graph

	epochs = 10
	batch_size = 1000
	window_size = 10

	with tf.Session(graph = train_graph) as sess:
		iteration = 1
		loss = 0
		sess.run(tf.global_variables_initializer())

		for e in range(1, epochs+1):
			batches = helper.get_batches(train_words, batch_size, window_size)
			start = time.time()

			for x,y in batches:
				feed = {inputs: x, 
						labels: np.array(y)[:, None]}
				train_loss, _ = sess.run([cost, optimizer], feed_dict = feed)
				loss += train_loss

				if iteration % 100 == 0:
					end = time.time()
					print("Epoch {}/{}".format(e, epochs),
						  "Iteration: {}".format(iteration),
						  "Avg. Training loss: {:.4f}".format(loss/100),
						  "{:.4f} sec/batch".format((end-start)/100))
					loss = 0
					start = time.time()

				if iteration % 1000 == 0:
					sim = similarity.eval()
					for i in range(valid_size):
						valid_word = int_to_vocab[valid_examples[i]]
						top_k = 8 # number of nearest neighbors
						nearest = (-sim[i, :]).argsort()[1:top_k+1]
						log = 'Nearest to %s:' % valid_word
						for k in range(top_k):
							close_word = int_to_vocab[nearest[k]]
							log = '%s %s,' % (log, close_word)
						print(log)
					
				iteration += 1

		save_path = saver.save(sess, "checkpoints/text8.ckpt")
		embed_mat = sess.run(normalized_embedding)

	return embed_mat

def visualize(embed_mat):

	viz_words = 500
	tsne = TSNE()
	embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

	fig, ax = plt.subplots(figsize = (14, 14))
	for idx in range(viz_words):
		plt.scatter(*embed_tsne[idx,:], color = "steelblue")
		plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha = 0.7)

def main():

	# Read in data
	os.chdir("/Users/MichaelChoie/Desktop/Data Science/deep-learning/embeddings")
	text = helper.read_data()

	# Process data
	words = helper.preprocess(text)
	vocab_to_int, int_to_vocab = helper.create_lookup_table(words)
	int_words = [vocab_to_int[word] for word in words]
	train_words = helper.subsampling(int_words)

	# Create directory for model checkpoints
	if not os.path.exists("checkpoints"):
		os.mkdirs("checkpoints")	

	# Build computational graph
	embed_mat = build_graph(train_graph, int_to_vocab, train_words)
	
	# Visualize network
	visualize(embed_mat)

if __name__ == "__main__":
	main()
