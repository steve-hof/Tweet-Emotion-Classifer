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
MAX_TWEET_LENGTH = 25
BATCH_SIZE = 24
LSTM_UNITS = 72
NUM_CLASSES = 2
ITERATIONS = 20000
LEARNING_RATE = 1e-4
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

	# has_emo = []
	# no_emo = []

	# for tweet in has_emotion:
	# 	has_emo.append(tokenize(tweet))
	# for tweet in no_emotion:
	# 	no_emo.append(tokenize(tweet))
	return has_emotion, no_emotion

def main():
	##################################################################
	### Load up the twitter GLOVE and split into words and vectors ###
	##################################################################
	glove_filepath = "../training_data/glove.twitter.27B/glove.twitter.27B.200d.txt"
	PAD_TOKEN = 0
	word2idx = {'PAD': PAD_TOKEN}
	weights = []

	with open(glove_filepath, 'r') as f:
		for index, line in enumerate(f):
			values = line.split()# word and weights separated by space
			word = values[0] #word is first symbol on each line
			word_weights = np.asarray(values[1:], dtype=np.float32)#remainder of line is weights for words
			word2idx[word] = index + 1 #PAD is zeroth index so shift by one
			weights.append(word_weights)

			if index + 1 == 10000:
			# Limit size for now
				break

	### Insert PAD weights at index 0.
	EMBEDDING_DIMENSION = len(weights[0])
	print(f"EMBEDDING_DIMENSION = {EMBEDDING_DIMENSION}")
	weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))
	
	### Append unknown and pad to end of vocab and initialize as random
	UNKNOWN_TOKEN = len(weights)
	word2idx['UNK'] = UNKNOWN_TOKEN
	weights.append(np.random.randn(EMBEDDING_DIMENSION))

	### Construct final vocab
	weights = np.asarray(weights)

	VOCAB_SIZE = weights.shape[0]
	print(f"vocab_size = {VOCAB_SIZE}")
	###############################################################
	### Import dataset for one emotion, clean and separate into ###	
	### has emotion and no emotion								###
	###############################################################
	
	dataset = pd.read_csv('../training_data/2018-E-c-En-train.txt', sep = '\t', quoting = 3, lineterminator = '\r')
	emotions = dataset.columns[2:]

	### For now we just use 'anger'
	emotion = 'anger'
	dataset = dataset[['Tweet', emotion]]
	total_num_tweets = len(dataset)

	has_emo_tweets, no_emo_tweets = separateOnEmo(dataset, emotion)
	num_has_emo, num_no_emo = len(has_emo_tweets), len(no_emo_tweets)
	total_num_tweets = num_has_emo + num_no_emo

	###################################################
	### Now we need to turn our tweets into vectors ###
	###################################################
	

	ids = np.zeros((total_num_tweets, MAX_TWEET_LENGTH), dtype='int32')
	tweetCounter = 0
	for tweet in has_emo_tweets:
		indexCounter = 0
		split = tweet.split()
		for word in split:
			try:
				ids[tweetCounter][indexCounter] = word2idx[word]
			except KeyError:
				ids[tweetCounter][indexCounter] = UNKNOWN_TOKEN 
			indexCounter += 1
			if indexCounter >= MAX_TWEET_LENGTH:
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
				if indexCounter >= MAX_TWEET_LENGTH:
					break
			tweetCounter += 1

if __name__ == '__main__':
	main()
	