#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import datetime
from os import listdir
from os.path import isfile, join
from random import randint
import pickle
import collections

FLAGS = re.MULTILINE | re.DOTALL


### Get and load the glove
def getGlove(filepath_glove):
	print("Loading GLOVE......")
	glove_vocab = []
	embedding_dict = {}
	embed_vector = []
	with open(filepath_glove, 'r', encoding='UTF-8') as f:
		for line in f.readlines():
			row = line.strip().split(' ')
			vocab_word = row[0]
			glove_vocab.append(vocab_word)
			embed_vector = [float(i) for i in row[1:]] #make list of word embeddings
			embedding_dict[vocab_word] = embed_vector
	print("GLOVE loaded")
	return glove_vocab, embed_vector, embedding_dict, len(glove_vocab), len(embed_vector)

def read_input_data(raw_text):
	content = raw_text
	content = content.split()
	content = np.array(content)
	print(f"old shape: {content.shape}")
	content = np.reshape(content, [-1, ])
	print(f"new shape: {content.shape}")
	return content

def getTrainingData(training_path):
	dataset = pd.read_csv(training_path, sep = '\t', quoting = 3, 
                   lineterminator = '\r')
	dataset = dataset.drop(['ID'], axis=1)
	dataset = dataset.dropna(axis = 0)
	n = 6838
	return dataset, n

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


#### Build dictionaries from the current training documents ####
#### Words are keys and unique integers are values ####
def build_dictionaries(words):
	print("Building tweet vector dictionaries......")
	#### Gives ordered list of word / count pairs
	count = collections.Counter(words).most_common()
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary) # (increases w/ each iter)
		#### rev_dict has unique integers as keys, words as values
		reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
	print("finished building dictionaries")
	return dictionary, reverse_dictionary

if __name__ == '__main__':
	training_path = 'training_data/2018-E-c-En-train.txt'
	glove_path = 'training_data/glove.twitter.27B.200d.txt'

	### Get tweet data
	dataset, n = getTrainingData(training_path)
	## make smaller version for now
	#dataset = dataset[:10]


	all_tweets_str = ' '.join(dataset['Tweet'].tolist())
	tokens = tokenize(all_tweets_str)

	### create content to embed with GLOVE ids
	tweets_for_glove = read_input_data(tokens)

	### Get GLOVE embeddings 
	glove_vocab, embed_vector, embedding_dict, glove_vocab_size, embedding_dim = getGlove(glove_path)

	#### number of words in the GLOVE file ####
	glove_vocab_size = len(glove_vocab)
	
	#### number of dimensions the embedding vectors have ####
	embedding_dim = len(embed_vector)
	dictionary, reverse_dictionary = build_dictionaries(tweets_for_glove)

	######################################################
	#### Create word embeddings that we'll load into  ####
	#### tensorflow. For each word in training data we ###
	#### search through embedding_dict. If word is    ####
	#### found we update, if not we assign small      ####
	#### random value that tf will update itself      ####
	######################################################

	doc_vocab_size = len(dictionary)
	dict_as_list = sorted(dictionary.items(), key = lambda x: x[1])
	embeddings_tmp = []

	print("Beginning to build word vectors........")
	for i in range(doc_vocab_size):
		if (i % 5000 == 0 and i != 0):
			print("still building word vector embeddings....")
		item = dict_as_list[i][0]
		if item in glove_vocab:
			embeddings_tmp.append(embedding_dict[item])
		else:
			rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
			embeddings_tmp.append(rand_num)
 
	# final embedding array corresponds to dictionary of words in the document
	embedding = np.asarray(embeddings_tmp)

	with open('training_data/wordVectors', 'wb') as f:
		pickle.dump(embedding, f)

	with open('training_data/wordsList', 'wb') as f:
		pickle.dump(dict_as_list, f)

