#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re
import os
import random
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from os import listdir
from os.path import isfile, join
from random import randint
import pickle
import collections
import random

dataset = pd.read_csv('../training_data/2018-E-c-En-train.txt', sep = '\t', quoting = 3, lineterminator = '\r')
emotions = dataset.columns[2:]
print("imported dataset")

print("creating w2v.......")
with open("../training_data/glove.twitter.27B/glove.twitter.27B.200d.txt") as lines:
	w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
			for line in lines}

print("finished creating w2v")
wordsList = list(w2v.keys())
print("wordsList is now a list")

print('saving wordsList.......')
with open('training_data/wordsList', 'wb') as f:
	pickle.dump(wordsList, f)
print("saved wordslist")


print("saving w2v.......")
with open('training_data/w2vec', 'wb') as fp:
	pickle.dump(w2v, fp)
print("saved w2v")



maxSeqLength = 10 #Maximum length of sentence
numDimensions = 200 #Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = word_list.index("i")
firstSentence[1] = word_list.index("thought")
firstSentence[2] = word_list.index("the")
firstSentence[3] = word_list.index("movie")
firstSentence[4] = word_list.index("was")
firstSentence[5] = word_list.index("incredible")
firstSentence[6] = word_list.index("and")
firstSentence[7] = word_list.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence)

with tf.Session() as sess:
	print(tf.nn.embedding_lookup(w2v,firstSentence).eval().shape)