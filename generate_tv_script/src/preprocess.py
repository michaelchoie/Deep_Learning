import numpy as np
import os 
import pickle 
from collections import Counter

def load_data(path):
	"""
	Load data from file
	"""

	file = os.path.join(path)
	with open(file, "r") as f:
		data = f.read()

	return data

def token_lookup():
	"""
	Generate a dictionary to turn punctuation into a token.
	:return: Tokenize dictionary where the key is the punctuation and the value is the token
	"""

	token_list = {
				 ".": "||Period||",
				 ",": "||Comma||",
				 '"': "|Quotation_Mark||",
				 ";": "||Semicolon||",
				 "!": "||Exclamation_Mark||",
				 "?": "||Question_Mark||",
				 "(": "||Left_Parentheses||",
				 ")": "||Right_Parentheses||",
				 "--": "||Dash||",
				 "\n": "||Return||"
				 }

	return token_list

def create_lookup_tables(text):
	"""
	Create lookup tables for vocabulary
	:param text: The text of tv scripts split into words
	:return: A tuple of dicts (vocab_to_int, int_to_vocab)
	"""
	word_counts = Counter(text)
	sorted_words = sorted(word_counts, key = word_counts.get, reverse = True)
	vocab_to_int = {word: ii for ii, word in enumerate(sorted_words)}
	int_to_vocab = {ii: word for ii, word in enumerate(sorted_words)}
	return (vocab_to_int, int_to_vocab)

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
	"""
	Preprocess text data
	:params dataset_path: path to data
	:params token_lookup: function
	:params create_lookup_tables: function
	"""

	# File starts with "[YEAR DATE 1989] © Twentieth Century Fox Film Corporation. All rights reserved."
	# Irrelevant information so remove
	text = load_data(dataset_path)
	text = text[81:]
	
	# Replace punctuation as tokens to make differentiating words and sentiment easier
	token_dict = token_lookup()
	for key, value in token_dict.items():
		text = text.replace(key, ' {} '.format(value))

	# Transform text for use in lookup tables
	text = text.lower()
	text = text.split()

	# Create lookup tables
	vocab_to_int, int_to_vocab = create_lookup_tables(text)
	int_text = [vocab_to_int[word] for word in text]

	pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open("preprocess.p", "wb"))

def load_preprocess():
	"""
	Load the Preprocessed Training data and return them in batches of <batch_size> or less	
	"""

	return pickle.load(open("preprocess.p", "rb"))


def save_params(params):
	"""
	Save parameters to file
	"""

	pickle.dump(params, open("params.p", "wb"))

def load_params():
	"""
	Load parameters from file
	"""

	return pickle.load(open("params.p", "rb"))

def get_batches(int_text, batch_size, seq_length):
	"""
	Return batches of input and target
	:param int_text: Text with the words replaced by their ids
	:param batch_size: The size of batch
	:param seq_length: The length of sequence
	:return: Batches as a Numpy array
	"""

	n_sequences = len(int_text) // (batch_size * seq_length) 
	n_words = n_sequences * batch_size * seq_length
	inputs = np.array(int_text[:n_words])

	input_batches = np.split(inputs.reshape(batch_size, -1), n_sequences, axis = 1)
	targets = np.array(int_text[1:n_words + 1])
	targets[-1] = int_text[0]
	target_batches = np.split(targets.reshape(batch_size, -1), n_sequences, axis = 1)

	return np.array(list(zip(input_batches, target_batches)))
