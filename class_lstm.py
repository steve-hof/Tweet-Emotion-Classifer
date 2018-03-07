#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re
import os
# from scipy import spacial
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
from os import listdir
from os.path import isfile, join
from random import randint
import collections
import random

FLAGS = re.MULTILINE | re.DOTALL
EMBEDDING_DIMENSION = 200
BATCH_SIZE = 24
NUM_CLASSES = 2
ITERATIONS = 4000
FLAGS = re.MULTILINE | re.DOTALL
"""
Clean tweets and create tags for often used
emojis
"""

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

def tokenize(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", '<hashtag>')
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    #text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def separateOnEmo(data, emotion):
	has_emotion = data[data[emotion] == 1]['Tweet']
	no_emotion = data[data[emotion] == 0]['Tweet']
	has_emo = []
	no_emo = []
	for tweet in has_emotion:
		has_emo.append(tokenize(tweet))
	for tweet in no_emotion:
		no_emo.append(tokenize(tweet))
	return has_emotion, no_emotion



class Model():

	def __init__(self, learning_rate, lstm_units, tweet_length):
		self.learning_rate = learning_rate
		self.lstm_units = lstm_units
		self.tweet_length = tweet_length

	def _getTrainBatch(self, has_emotion, no_emotion, ids):
		labels = []
		total = has_emotion + no_emotion
		test_ratio = 0.2
		num_test = int(total * test_ratio)
		split_amount = int(num_test * 0.5)
		arr = np.zeros([BATCH_SIZE, self.tweet_length])
		for i in range(BATCH_SIZE):
			if (i % 2 == 0): 
				num = randint(1, has_emotion - split_amount)
				labels.append([1,0])
			else:
				num = randint(has_emotion + split_amount, total)
				labels.append([0,1])
			arr[i] = ids[num-1:num]
		return arr, labels

	def _getTestBatch(self, has_emotion, no_emotion, ids):
		labels = []
		total = has_emotion + no_emotion
		test_ratio = 0.2
		num_test = int(total * test_ratio)
		split_amount = int(num_test * 0.5)
		arr = np.zeros([BATCH_SIZE, self.tweet_length])
		for i in range(BATCH_SIZE):
			num = randint(has_emotion - split_amount + 1, has_emotion + split_amount - 1)
			if (num <= has_emotion):
				labels.append([1,0])
			else:
				labels.append([0,1])
			arr[i] = ids[num-1:num]
		return arr, labels

	def _getIDs(self):
			ids = np.zeros((total_num_tweets, self.tweet_length), dtype='int32')
		tweetCounter = 0
		for tweet in has_emo_tweets:
			indexCounter = 0
			split = tweet.split()
			for word in split:
				try:
					ids[tweetCounter][indexCounter] = word2idx[word]
				except KeyError:
					ids[tweetCounter][indexCounter] = word2idx['UNK'] 
				indexCounter += 1
				if indexCounter >= self.tweet_length:
					break
			tweetCounter += 1

		for tweet in no_emo_tweets:
				indexCounter = 0
				split = tweet.split()
				for word in split:
					try:
						ids[tweetCounter][indexCounter] = word2idx[word]
					except KeyError:
						ids[tweetCounter][indexCounter] = UNKNOWN_TOKEN 
					indexCounter += 1
					if indexCounter >= self.tweet_length:
						break
				tweetCounter += 1


	def train(has):
		tf.reset_default_graph()
			
		### Set up placeholders for input and labels
		with tf.name_scope("Labels") as scope:
			labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
		with tf.name_scope("Input") as scope:
			input_data = tf.placeholder(tf.int32, [BATCH_SIZE, self.tweet_length])

		### Get embedding vector
		with tf.name_scope("Embeds_Layer") as scope:
			embedding = tf.Variable(tf.zeros([BATCH_SIZE, self.tweet_length, EMBEDDING_DIMENSION]),dtype=tf.float32, name='embedding')
			embed = tf.nn.embedding_lookup(weights, input_data)

		### Set up LSTM cell then wrap cell in dropout layer to avoid overfitting
		with tf.name_scope("LSTM_Cell") as scope:
			lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
			lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

		### Combine the 3D input with the LSTM cell and set up network
		### 'value' is the last hidden state vector
		with tf.name_scope("RNN_Forward") as scope:
			value, _ = tf.nn.dynamic_rnn(lstmCell, embed, dtype=tf.float32)

		with tf.name_scope("Fully_Connected") as scope:
			weight = tf.Variable(tf.truncated_normal([self.lstm_units, NUM_CLASSES]), name='weights')
			bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='bias')
			value = tf.transpose(value, [1, 0, 2], name='last_lstm')
			last = tf.gather(value, int(value.get_shape()[0]) - 1)
			tf.summary.histogram("weights", weight)
			tf.summary.histogram("biases", bias)


		with tf.name_scope("Predictions") as scope:
			prediction = (tf.matmul(last, weight) + bias)

		# with tf.name_scope("Prediction / Accuracy") as scope:
			

		### Cross entropy loss with a softmax layer on top
		### Using Adam for optimizer with 0.0001 learning rate
		with tf.name_scope("Loss_and_Accuracy") as scope:
			correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
			accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

		with tf.name_scope("Training") as scope:
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
		
		sess = tf.InteractiveSession()
		saver = tf.train.Saver() #(tf.global_variables())
		sess.run(tf.global_variables_initializer())
		
		### Output directory for models and summaries
		timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		logdir = os.path.abspath(os.path.join(os.path.curdir, "all_emo_board", timestamp))
		print(f"Writing to {logdir}")

		### Summaries for loss and accuracy
		# loss_summary = tf.summary.scalar("Loss", loss)
		acc_summary = tf.summary.scalar('Accuracy', accuracy)

		### Training summaries
		# train_summary_op = tf.summary.merge([loss_summary, acc_summary])
		train_summary_dir = os.path.join(logdir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)		

		### Testing summaries
		# test_summary_op = tf.summary.merge([loss_summary, acc_summary])
		test_summary_dir = os.path.join(logdir, "summaries", "test")
		test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
		
		summary_op = tf.summary.merge_all()
		# summary_op = tf.summary.merge([train_summary_op, test_summary_op])
		# Checkpointing
		checkpoint_dir = os.path.abspath(os.path.join(logdir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		# Tensorflow assumes this directory already exists so we need to create it
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		


		for i in range(ITERATIONS):
			#Next Batch of reviews
			nextTrainBatch, nextTrainBatchLabels = self._getTrainBatch(num_has_emo, num_no_emo, ids);
			sess.run(optimizer, {input_data: nextTrainBatch, labels: nextTrainBatchLabels})

			#Write training summary to board
			if (i % 50 == 0):
				summary = sess.run(summary_op, {input_data: nextTrainBatch, labels: nextTrainBatchLabels})
				train_summary_writer.add_summary(summary, i)
				train_summary_writer.flush()

				nextTestBatch, nextTestBatchLabels = self._getTestBatch(num_has_emo, num_no_emo, ids);
				testSummary = sess.run(summary_op, {input_data: nextTestBatch, labels: nextTestBatchLabels})
				test_summary_writer.add_summary(testSummary, i)
				test_summary_writer.flush()
			
			#Save network every so often
			if (i % 1000 == 0 and i != 0):
				save_path = saver.save(sess, f"all_emo_models/{emotion}_pretrained_lstm.ckpt", global_step=i)
				print(f"saved to {save_path}")
			


		train_summary_writer.close()
		test_summary_writer.close()


def main():

	##################################################################
	### Load up the twitter GLOVE and split into words and vectors ###
	##################################################################
	glove_filepath = "training_data/glove.twitter.27B/glove.twitter.27B.200d.txt"
	PAD_TOKEN = 0
	word2idx = {'PAD': PAD_TOKEN}
	weights = []

	with open(glove_filepath, 'r', encoding='UTF-8') as f:
		for index, line in enumerate(f):

			values = line.split()# word and weights separated by space
			word = values[0] #word is first symbol on each line
			word_weights = np.asarray(values[1:], dtype=np.float32)#remainder of line is weights for words
			if len(word_weights) != 200:
				continue
			word2idx[word] = index + 1 #PAD is zeroth index so shift by one
			weights.append(word_weights)
			if len(word_weights) != 200:
				print(f"fucking up at index {index}")
				print(f"index {index} is length {len(word_weights)}")

			if index + 1 == 200000:
			# Limit size for now
				break

	### Insert PAD weights at index 0.
	EMBEDDING_DIMENSION = len(weights[0])
	weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))
	
	### Append unknown and pad to end of vocab and initialize as random
	UNKNOWN_TOKEN = len(weights)
	word2idx['UNK'] = UNKNOWN_TOKEN
	weights.append(np.random.randn(EMBEDDING_DIMENSION))

	### Construct final vocab
	weights = np.asarray(weights, dtype=np.float32)
	VOCAB_SIZE = weights.shape[0]



	###############################################################
	### Import dataset for one emotion, clean and separate into ###	
	### has emotion and no emotion								###
	###############################################################
	
	dataset = pd.read_csv('training_data/2018-E-c-En-train.txt', sep = '\t', quoting = 3, lineterminator = '\r')
	emotions = dataset.columns[2:]

	### For now we just use 'anger'
	emotion = 'anger'
	dataset = dataset[['Tweet', emotion]]
	total_num_tweets = len(dataset)

	has_emo_tweets, no_emo_tweets = separateOnEmo(dataset, emotion)
	num_has_emo, num_no_emo = len(has_emo_tweets), len(no_emo_tweets)
	total_num_tweets = num_has_emo + num_no_emo

	##############################################################################
	### Now we need to turn our tweets into vectors representing their indices ###
	##############################################################################



	print(f"ids shape: {ids.shape}")


	nn = Model(learning_rate, )
	
if __name__ == '__main__':
	main()
	