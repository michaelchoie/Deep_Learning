# -*- coding: utf-8 -*-

import time
import sys
from collections import Counter
import numpy as np

class SentimentNetwork:
    def __init__(self, reviews, labels, min_count = 10, polarity = 0.1, hidden_nodes = 10, learning_rate = 0.1):
        """
        Create a SentimentNetwork with the given settings
        """

        np.random.seed(1)
        self.pre_process_data(reviews, labels, min_count, polarity)
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity):
        """
        Convert text data into numerical form
        """
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if (labels[i] == "positive\n"):
                for word in reviews[i].split(' '):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(' '):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term, cnt in list(total_counts.most_common()):
            if (cnt >= 50):
                pos_neg_ratio = positive_counts[term] / (1 + negative_counts[term])
                pos_neg_ratios[term] = pos_neg_ratio

        for word, ratio in pos_neg_ratios.most_common():
            if (ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        
        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if (total_counts[word] > min_count):
                    if (word in pos_neg_ratios.keys()):
                        if ((pos_neg_ratios[word] >= polarity) or (pos_neg_ratios[word] <= -polarity)):
                            review_vocab.add(word)
                else: 
                    review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # Populate label_vocab with all of the words in the given labels
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        Initialize network with nodes, weights, and learning rate
        """
        
        # Set number of nodes in input, hidden and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    def get_target_for_label(self,label):
        """
        Map discrete values to binary scale
        """
        
        if(label == 'positive\n'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        """
        Create formula for sigmoid activation function
        """

        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        """
        Create formula for sigmoid derivative
        """

        return output * (1 - output)
    
    def train(self, training_reviews_raw, training_labels):
        """
        Train neural network and update weights
        """
        
        # Preprocess reviews
        training_reviews = list()
        for review in training_reviews_raw:
            indicies = set()            
            for word in review.split(' '):
                if (word in self.word2index.keys()):
                    indicies.add(self.word2index[word])
            training_reviews.append(list(indicies))
        
        # Make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions
        correct_so_far = 0

        # Keep track of time
        start = time.time()
        
        # Perform feedforward/backpropagation
        for i in range(len(training_reviews)):
            
            # Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # Feedforward
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
            
            # Backpropagation 
            layer_2_error = layer_2 - self.get_target_for_label(label) 
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) 
            layer_1_delta = layer_1_error 

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate 
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate 

            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'positive\n'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'negative\n'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print('')
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of correct predictions and time taken to make
        correct = 0
        start = time.time()

        # Loop through each of the given reviews and call run to predict 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out prediction accuracy and speed 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        
        # Hidden layer
        self.layer_1 *= 0
        unique_indicies = set()
        for word in review.lower().split(' '):
            if word in self.word2index.keys():
                unique_indicies.add(self.word2index[word])
        for index in unique_indicies:
            self.layer_1 += self.weights_0_1[index]

        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        # Return POSITIVE/NEGATIVE if meets prediction value threshold
        if(layer_2[0] >= 0.5):
            return "positive\n"
        else:
            return "negative\n"
