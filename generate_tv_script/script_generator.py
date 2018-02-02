import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
import preprocess

def get_inputs():
	"""
	Create TF Placeholders for input, targets, and learning rate.
	:return: Tuple (input, targets, learning rate)
	"""

	inputs = tf.placeholder(tf.int32, [None, None], name = "input")
	targets = tf.placeholder(tf.int32, [None, None], name = "targets")
	learning_rate = tf.placeholder(tf.float32, name = "learning_rate")

	return (inputs, targets, learning_rate)

def get_init_cell(batch_size, rnn_size):
	"""
	Create an RNN Cell and initialize it.
	:param batch_size: Size of batches
	:param rnn_size: Size of RNNs
	:return: Tuple (cell, initialize state)
	"""

	lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
	initial_state = cell.zero_state(batch_size, tf.float32)
	initial_state = tf.identity(initial_state, name = "initial_state")

	return (cell, initial_state)

def get_embed(input_data, vocab_size, embed_dim):
	"""
	Create embedding for <input_data>.
	:param input_data: TF placeholder for text input.
	:param vocab_size: Number of words in vocabulary.
	:param embed_dim: Number of embedding dimensions
	:return: Embedded input.
	"""

	embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
	embed = tf.nn.embedding_lookup(embedding, input_data)

	return embed

def build_rnn(cell, inputs):
	"""
	Create a RNN using a RNN Cell
	:param cell: RNN Cell
	:param inputs: Input text data
	:return: Tuple (Outputs, Final State)
	"""

	outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
	final_state = tf.identity(final_state, name = "final_state")

	return (outputs, final_state)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
	"""
	Build part of the neural network
	:param cell: RNN cell
	:param rnn_size: Size of rnns
	:param input_data: Input data
	:param vocab_size: Vocabulary size
	:param embed_dim: Number of embedding dimensions
	:return: Tuple (Logits, FinalState)
	"""

	embed = get_embed(input_data, vocab_size, embed_dim)
	outputs, final_state = build_rnn(cell, embed)
	logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn = None)

	return (logits, final_state)

def get_tensors(loaded_graph):
	"""
	Get input, initial state, final state, and probabilities tensor from <loaded_graph>
	:param loaded_graph: TensorFlow graph loaded from file
	:return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
	"""
	input_tensor = loaded_graph.get_tensor_by_name("input:0")
	initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
	final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
	probs_tensor = loaded_graph.get_tensor_by_name("probs:0")

	return (input_tensor, initial_state_tensor, final_state_tensor, probs_tensor)

def pick_word(probabilities, int_to_vocab):
	"""
	Pick the next word in the generated text
	:param probabilities: Probabilites of the next word
	:param int_to_vocab: Dictionary of word ids as the keys and words as the values
	:return: String of the predicted word
	"""
	
	return int_to_vocab[np.random.choice(np.arange(len(probabilities)), p = probabilities)]

def train_neural_net(int_text, int_to_vocab):

	# Define hyperparameters
	num_epochs = 50
	batch_size = 128
	rnn_size = 1024
	embed_dim = 512
	seq_length = 16
	learning_rate = 0.001
	show_every_n_batches = 11

	# Define path to checkpoint files
	save_dir = './save'	

	# Build the computational graph
	train_graph = tf.Graph()
	with train_graph.as_default():

		# Load tensors
		vocab_size = len(int_to_vocab)
		input_text, targets, lr = get_inputs()
		input_data_shape = tf.shape(input_text)
		cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
		logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

		# Probabilities for generating words
		probs = tf.nn.softmax(logits, name = "probs")

		# Loss function & Optimizer
		cost = seq2seq.sequence_loss(logits,
									 targets,
									 tf.ones([input_data_shape[0], input_data_shape[1]]))
		optimizer = tf.train.AdamOptimizer(lr)

		# Initialize clipped gradients
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.apply_gradients(capped_gradients)

	# Create batches
	batches = preprocess.get_batches(int_text, batch_size, seq_length)

	# Train the neural network
	with tf.Session(graph = train_graph) as sess:
		sess.run(tf.global_variables_initializer())
        
		for epoch in range(num_epochs):
			state = sess.run(initial_state, {input_text: batches[0][0]})

			for batch, (x,y) in enumerate(batches):
				feed = {input_text: x,
						targets: y,
						initial_state: state,
						lr: learning_rate}
				train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

			# Print results
			if (epoch * len(batches) + batch) % show_every_n_batches == 0:
				print("Epoch {:>3} Batch {:>4}/{}	train_loss = {:.3f}".format(epoch, 
																				batch, 
																				len(batches), 
																				train_loss))

		# Save model
		saver = tf.train.Saver()
		saver.save(sess, save_dir)
		print("Model trained and saved")

	# Save parameters for checkpoint
	helper.save_params((seq_length, save_dir))

def main():
	
	# Load data
	data_dir = "/Users/MichaelChoie/Desktop/Data Science/deep-learning/tv-script-generation/data/simpsons/moes_tavern_lines.txt"
	text = preprocess.load_data(data_dir)
	preprocess.preprocess_and_save_data(data_dir, preprocess.token_lookup, preprocess.create_lookup_tables)
	int_text, vocab_to_int, int_to_vocab, token_dict = preprocess.load_preprocess()

	# Train neural network
	train_neural_net(int_text, int_to_vocab)

	# Load checkpoint file and generate TV script
	gen_length = 200
	prime_word = 'homer_simpson'

	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
	    # Load saved model
	    loader = tf.train.import_meta_graph(load_dir + '.meta')
	    loader.restore(sess, load_dir)

	    # Get Tensors from loaded model
	    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

	    # Sentences generation setup
	    gen_sentences = [prime_word + ':']
	    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

	    # Generate sentences
	    for n in range(gen_length):
	        # Dynamic Input
	        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
	        dyn_seq_length = len(dyn_input[0])

	        # Get Prediction
	        probabilities, prev_state = sess.run(
	            [probs, final_state],
	            {input_text: dyn_input, initial_state: prev_state})
	        
	        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

	        gen_sentences.append(pred_word)
	    
	    # Remove tokens
	    tv_script = ' '.join(gen_sentences)
	    for key, token in token_dict.items():
	        ending = ' ' if key in ['\n', '(', '"'] else ''
	        tv_script = tv_script.replace(' ' + token.lower(), key)
	    tv_script = tv_script.replace('\n ', '\n')
	    tv_script = tv_script.replace('( ', '(')
	        
	    print(tv_script)

if __name__ == "__main__":
	main()
