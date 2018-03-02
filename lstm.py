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
BATCH_SIZE = 50
LSTM_UNITS = 64
NUM_CLASSES = 2
ITERATIONS = 10000
FLAGS = re.MULTILINE | re.DOTALL

class Model():

	def __init__(self, wordVectors, wordsList):
		self.max_dimensions = MAX_DIMENSIONS
		self.max_tweet_length = MAX_TWEET_LENGTH
		self.batchSize = BATCH_SIZE
		self.test_batchSize = BATCH_SIZE
		self.lstmUnits = LSTM_UNITS
		self.numCLasses = NUM_CLASSES
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
		# IDs = self._getIDs(hasEmotionTweets, noEmotionTweets)

		return num_has_emotion, num_no_emotion#, IDs

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



if __name__ == '__main__':
	with open ('training_data/wordVectors', 'rb') as f:
		wordVectors = pickle.load(f)

	with open ('training_data/wordsList', 'rb') as f:
		wordsList_vec = pickle.load(f)

	wordsList = []
	for word in wordsList_vec:
		wordsList.append(str(word[0]))


	training_file = 'training_data/2018-E-c-En-train.txt'
	dataset = pd.read_csv(training_file, sep = '\t', quoting = 3, 
										lineterminator = '\r')
	emotions = dataset.columns.tolist()[2:]

	nn = Model(wordVectors, wordsList)
	for emotion in emotions:
	### add ids to return values later
		num_has_emotion, num_no_emotion = nn.prepareData(dataset, emotion)



