"""
Provide data loading/processing functions 
"""

import numpy as np
from urllib.request import urlretrieve
import os
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import re
import random
from collections import Counter

class DLProgress(tqdm):
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num

def read_data():

	dataset_folder_path = 'data'
	dataset_filename = 'text8.zip'
	dataset_name = 'Text8 Dataset'

	if not isfile(dataset_filename):
		with DLProgress(unit="B", unit_scale=True, miniters=1, desc=dataset_name) as pbar:
			urlretrieve('http://mattmahoney.net/dc/text8.zip', dataset_filename, pbar.hook)

	if not isdir(dataset_folder_path):
		with zipfile.Zipfile(dataset_filename) as zip_ref:
			zip_ref.extractall(dataset_folder_path)

	with open("data/text8") as f:
		text = f.read()

	return text

def preprocess(text):
	"""
	Replace punctuation with tokens and remove rare words
	:param text:
	:return trimmed_words:
	"""

	text = text.lower()
	text = text.replace('.', ' <PERIOD> ')
	text = text.replace(',', ' <COMMA> ')
	text = text.replace('"', ' <QUOTATION_MARK> ')
	text = text.replace(';', ' <SEMICOLON> ')
	text = text.replace('!', ' <EXCLAMATION_MARK> ')
	text = text.replace('?', ' <QUESTION_MARK> ')
	text = text.replace('(', ' <LEFT_PAREN> ')
	text = text.replace(')', ' <RIGHT_PAREN> ')
	text = text.replace('--', ' <HYPHENS> ')
	text = text.replace('?', ' <QUESTION_MARK> ')
	text = text.replace(':', ' <COLON> ')

	words = text.split()

	word_counts = Counter(words)
	trimmed_words = [word for word in words if word_counts[word] > 5]

	return trimmed_words

def create_lookup_table(words):
	"""
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....	
	"""
	word_counts = Counter(words)
	sorted_vocab = sorted(word_counts, key = word_counts.get, reverse = True)
	int_to_vocab = {ii: word for word, ii in enumerate(sorted_vocab)}
	vocab_to_int = {word: ii for word, ii in int_to_vocab.items()}

	return vocab_to_int, int_to_vocab

def subsampling(int_words):
	"""
	Randomly remove words more frequent than threshold with some probability 
	Akin to removing stop words to reduce noise
	Probability(word) = 1 - sqrt(threshold / frequency(word))
	:param int_words: list of numbers corresponding to words in text
	:return train_words:
	"""

	threshold = 1e-5 # standard recommended value
	word_counts = Counter(int_words)
	total_count = len(int_words)
	freqs = {word: count/total_count for word, count in word_counts.items()}
	p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
	train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

	return train_words

def get_target(words, idx, window_size=5):
    """ 
    Get a list of words in a window (number R inbetween [1:window_size]) around an index
    :param words: words in a batch
    :param idx: index of word
    :param window_size: how many surrounding words to look at
    :return target_words: target words in the window
    """

    R = np.random.randint(1, window_size+1) # randint isn't inclusive so +1
    start = idx - R if (idx - R) > 0 else 0 # handle edge case
    stop = idx + R
    target_words = set(words[start:idx] + words[idx+1:stop+1])

    return target_words

def get_batches(words, batch_size, window_size=5):
	"""
    Create generator of word batches as a tuple (inputs, targets)
    :param words: text
    :param batch_size: The size of batch
    :param window_size: The length of sequence
	"""

	n_batches = len(words) // batch_size
	words = words[:n_batches * batch_size]

	for idx in range(0, len(words), batch_size):
		x, y = [], []
		batch = words[idx:idx + batch_size]
		for ii in range(len(batch)):
			batch_x = batch[ii]
			batch_y = get_target(batch, ii, window_size)
			y.extend(batch_y)
			x.extend([batch_x] * len(batch_y))
		yield x,y

