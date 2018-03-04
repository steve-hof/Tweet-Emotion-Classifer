#!/usr/bin/env python3
import collections
import random
from scipy import spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import datetime
from tensorflow.contrib import rnn
from os import listdir
from os.path import isfile, join
from random import randint
import pickle

MAX_DIMENSIONS = 200
MAX_TWEET_LENGTH = 25
BATCH_SIZE = 24
LSTM_UNITS = 6
NUM_CLASSES = 2
ITERATIONS = 1000
FLAGS = re.MULTILINE | re.DOTALL

class Model():

	def __init__(self, wordVectors, wordsList):
		self.max_dimensions = MAX_DIMENSIONS
		self.max_tweet_length = MAX_TWEET_LENGTH
		self.batchSize = BATCH_SIZE
		self.testBatchSize = BATCH_SIZE
		self.lstmUnits = LSTM_UNITS
		self.numClasses = NUM_CLASSES
		self.iterations = ITERATIONS
		self.wordsList = wordsList
		self.wordVectors = wordVectors

	def prepareData(self, dataset, emotion):
		dataset = dataset.drop(['ID'], axis=1)
		dataset = dataset.dropna(axis = 0)
		hasEmotionTweets, noEmotionTweets = self.__getEmotions(dataset, emotion)

		hasEmotionTweets = self.__cleanTweets(hasEmotionTweets)
		noEmotionTweets = self.__cleanTweets(noEmotionTweets)
		num_has_emotion = len(hasEmotionTweets)
		num_no_emotion = len(noEmotionTweets)
		IDs = self._getIDs(hasEmotionTweets, noEmotionTweets)

		return num_has_emotion, num_no_emotion, IDs

	def __getEmotions(self, dataset, emotion):
		has_emotion = dataset[dataset[emotion] == 1][['Tweet', emotion]]
		no_emotion = dataset[dataset[emotion] == 0][['Tweet', emotion]]
		return has_emotion, no_emotion

	def __cleanTweets(self, X):
		n = len(X)
		clean_tweets = []
		X = X['Tweet'].tolist()
		for tweet in X:
			clean_tweets.append(self.tokenize(tweet))
		return clean_tweets

	def hashtag(self, text):
	    text = text.group()
	    hashtag_body = text[1:]
	    if hashtag_body.isupper():
	        result = " {} ".format(hashtag_body.lower())
	    else:
	        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
	    return result

	def allcaps(self, text):
	    text = text.group()
	    return text.lower() + " <allcaps>"

	def tokenize(self, text):
	    eyes = r"[8:=;]"
	    nose = r"['`\-]?"
	    
	    def re_sub(self, pattern, repl):
	        return re.sub(pattern, repl, text, flags=FLAGS)
	    
	    text = re_sub(self, r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
	    text = re_sub(self, r"@\w+", "<user>")
	    text = re_sub(self, r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
	    text = re_sub(self, r"{}{}p+".format(eyes, nose), "<lolface>")
	    text = re_sub(self, r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
	    text = re_sub(self, r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
	    text = re_sub(self, r"/"," / ")
	    text = re_sub(self, r"<3","<heart>")
	    text = re_sub(self, r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
	    text = re_sub(self, r"#\S+", '<hashtag>')
	    text = re_sub(self, r"([!?.]){2,}", r"\1 <repeat>")
	    text = re_sub(self, r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
	    
	    #text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
	    text = re_sub(self, r"([A-Z]){2,}", self.allcaps)

	    return text.lower()

	def _getIDs(self, hasEmotionTweets, noEmotionTweets):
		numTweets = len(hasEmotionTweets) + len(noEmotionTweets)
		tweetCounter = 0
		ids = np.zeros((numTweets, self.max_tweet_length), dtype='int32')
		print("Please be patient while we play with massive matrices...")
		for tweet in hasEmotionTweets:
			index = 0
			split = tweet.split()
			for word in split:
				try:
					ids[tweetCounter][index] = self.wordsList.index(word)
				except ValueError:
					ids[tweetCounter][index] = 399999 #Vector for unknown words
				index += 1
				if index >= self.max_tweet_length:
					break
			tweetCounter += 1

		for tweet in noEmotionTweets:
			index = 0
			split = tweet.split()
			for word in split:
				try:
					ids[tweetCounter][index] = self.wordsList.index(word)
				except ValueError:
					ids[tweetCounter][index] = 399999 #Vector for unknown words
				index += 1
				if index >= self.max_tweet_length:
					break
			tweetCounter += 1

		return ids

	def _getTrainBatch(self, has_emotion, no_emotion):
		labels = []
		total = has_emotion + no_emotion
		test_ratio = 0.2
		num_test = int(total * test_ratio)
		split_amount = int(num_test * 0.5)
		arr = np.zeros([self.batchSize, self.max_tweet_length])
		for i in range(self.batchSize):
			if (i % 2 == 0): 
				num = randint(1, has_emotion - split_amount)
				labels.append([1,0])
			else:
				num = randint(has_emotion + split_amount, total)
				labels.append([0,1])
			arr[i] = ids[num-1:num]
		return arr, labels

	def _getTestBatch(self, has_emotion, no_emotion):
		labels = []
		total = has_emotion + no_emotion
		test_ratio = 0.2
		num_test = int(total * test_ratio)
		split_amount = int(num_test * 0.5)
		arr = np.zeros([self.testBatchSize, self.max_tweet_length])
		for i in range(self.testBatchSize):
			num = randint(has_emotion - split_amount + 1, has_emotion + split_amount - 1)
			if (num <= has_emotion):
				labels.append([1,0])
			else:
				labels.append([0,1])
			arr[i] = ids[num-1:num]
		return arr, labels

	def trainNet(self, num_has_emotion, num_no_emotion, emotion,):
		tf.reset_default_graph()
		
		### Set up placeholders for input and labels
		with tf.name_scope("Labels") as scope:
			labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
		with tf.name_scope("Input") as scope:
			input_data = tf.placeholder(tf.int32, [self.batchSize, self.max_tweet_length])

		### Get embedding vector
		with tf.name_scope("Embeddings") as scope:
			data = tf.Variable(tf.zeros([self.batchSize, self.max_tweet_length, self.max_dimensions]),dtype=tf.float32)
			data = tf.nn.embedding_lookup(self.wordVectors, input_data)

		### Set up LSTM cell then wrap cell in dropout layer to avoid overfitting
		with tf.name_scope("LSTM_Cell") as scope:
			lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
			lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

		### Combine the 3D input with the LSTM cell and set up network
		### 'value' is the last hidden state vector
		with tf.name_scope("RNN_Forward") as scope:
			value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

		with tf.name_scope("Output_Layer") as scope:
			weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]), name='weights')
			bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]), name='bias')
			value = tf.transpose(value, [1, 0, 2], name='last_lstm')
			last = tf.gather(value, int(value.get_shape()[0]) - 1)
			tf.summary.histogram("weights", weight)

		with tf.name_scope("Predictions") as scope:
			prediction = (tf.matmul(last, weight) + bias)

		# with tf.name_scope("Prediction / Accuracy") as scope:
			

		### Cross entropy loss with a softmax layer on top
		### Using Adam for optimizer with 0.0001 learning rate
		with tf.name_scope("Loss_and_Accuracy") as scope:
			correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
			accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
			tf.summary.scalar("Training Loss", loss)
			tf.summary.scalar('Training Accuracy', accuracy)

		with tf.name_scope("Training") as scope:
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
			

		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		### Tensorboard set-up
		# tf.summary.scalar('Training Loss', loss)
		# tf.summary.scalar('Training Accuracy', accuracy)
		# tf.summary.scalar('Testing Accuracy', test_accuracy)
		merged = tf.summary.merge_all()
		logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
		writer = tf.summary.FileWriter(logdir, sess.graph)

		for i in range(self.iterations):
			#Next Batch of reviews
			nextBatch, nextBatchLabels = self._getTrainBatch(num_has_emotion, num_no_emotion);
			sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

			#Write summary to board
			if (i % 20 == 0):
				summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
				writer.add_summary(summary, i)

			#Save network every so often
			if (i % 200 == 0 and i != 0):
				save_path = saver.save(sess, f"models/{emotion}_pretrained_lstm.ckpt", global_step=i)
				print(f"saved to {save_path}")
		writer.close()

		### testing
		iterations = 20
		counter = 0
		accuracies = []
		for i in range(iterations):
			nextBatch, nextBatchLabels = self._getTestBatch(num_has_emotion, num_no_emotion);
			print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
			accuracies.append(sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}))
		average_accuracy = np.asarray(accuracies).mean()
		print(f"Average Accuracy: {average_accuracy}")
		with open('results/accuracy.txt', 'a') as f:
			print(f"\nAverage Test Accuarcy for {emotion}: {average_accuracy}\n", file=f)


if __name__ == '__main__':
	with open ('training_data/wordVectors', 'rb') as f:
		wordVectors = pickle.load(f)
	wordVectors = wordVectors.astype('float32')

	with open ('training_data/wordsList', 'rb') as f:
		wordsList_vec = pickle.load(f)

	wordsList = []
	for word in wordsList_vec:
		wordsList.append(str(word[0]))


	training_file = 'training_data/2018-E-c-En-train.txt'
	dataset = pd.read_csv(training_file, sep = '\t', quoting = 3, 
										lineterminator = '\r')
	# small_dataset = dataset[0:10]
	emotions = dataset.columns.tolist()[2:]


	nn = Model(wordVectors, wordsList)
	# for emotion in emotions:
	### add ids to return values later
	emotion = 'joy'
	num_has_emotion, num_no_emotion, ids = nn.prepareData(dataset, emotion)

	nn.trainNet(num_has_emotion, num_no_emotion, emotion)
