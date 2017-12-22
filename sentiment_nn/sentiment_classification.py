#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import Counter
import numpy as np
from sentiment_network import SentimentNetwork

os.chdir("/Users/MichaelChoie/Desktop/Data Science/deep-learning/sentiment-network")

def pretty_print_review_and_label(reviews, labels, i):
	print(labels[i] + '\t:\t' + reviews[i][:80] + '...')

def test_pretty_print(reviews, labels):
    print('labels.txt \t : \t reviews.txt')
    pretty_print_review_and_label(reviews, labels, 2137)
    pretty_print_review_and_label(reviews, labels, 9)

def open_close_file(file):
	with open(file, 'r') as g:
		output = g.readlines()
	return output

def test_neural_net(mlp, reviews, labels):
    mlp.train(reviews[:-1000], labels[:-1000])
    print('\n')
    mlp.test(reviews[-1000:], labels[-1000:])
    print('\n')

def get_most_similar_words(mlp, focus = 'horrible'):
    most_similar = Counter()
    
    for word in mlp.word2index.keys():
        most_similar[word] = np.dot(mlp.weights_0_1[mlp.word2index[word]], 
                                    mlp.weights_0_1[mlp.word2index[focus]])
    
    print([x[0] for x in most_similar.most_common()[:10]])

def main():
    print('\nLoading the text data')
    reviews = open_close_file('reviews.txt')
    labels = open_close_file('labels.txt')

    print('\nLoading the sentiment network')
    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate = 0.001)
    
    test_pretty_print(reviews, labels)
    test_neural_net(mlp, reviews, labels)
    get_most_similar_words(mlp, 'excellent')
    get_most_similar_words(mlp, 'terrible')


if __name__ == '__main__':
    main()