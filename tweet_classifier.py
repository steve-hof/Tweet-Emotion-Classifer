#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:47:23 2018

@author: stevehof
"""

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle

# Get tweet data
df = pd.read_csv('training_data/2018-E-c-En-train.txt', sep='\t', quoting=3,
                 lineterminator='\r')

df = df.drop(['ID'], axis=1)
df = df.dropna(axis=0)
n = 6838

# Cleaning the tweets
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

clean_tweets = []

for i in range(0, n):
    tweet = re.sub('[^a-zA-Z]', ' ', df['Tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    # tweet = [word for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    clean_tweets.append(tweet)

# save clean_tweets to file
with open('clean_tweets', 'wb') as f:
    pickle.dump(clean_tweets, f)

# load saved clean_tweets
# with open ('clean_tweets', 'rb') as f:
#    clean_tweets = pickle.load(f)

# CREATE BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(clean_tweets).toarray()


# y = df.iloc[0:n, 1].values


class Emotion:
    def __init__(self, name, col):
        self.name = name
        self.data = df.iloc[0:n, col].values

    def get_data(self):
        return self.data

    def get_name(self):
        return self.name


emotion_list = []

labels = {'anger': 1, 'anticipation': 2, 'disgust': 3, 'fear': 4, 'joy': 5, 'love': 6, 'optimism': 7, 'pessimism': 8,
          'sadness': 9, 'surprise': 10, 'trust': 11}

for name, data in labels.items():
    emotion_list.append(Emotion(name, data))

# For each emotion, split the df into the Training set and Test set and run algorithms
with open('tweet_accuracy_further.txt', 'w') as f:
    for emotion in emotion_list:
        print(f"Classification accuracy for {emotion.get_name()}", file=f)
        X_train, X_test, y_train, y_test = train_test_split(X, emotion.get_data(), test_size=0.2, random_state=0)

        # Run through each classifier, train on X_train and y_train and the test them using the score function
        algs = [
            svm.SVC(),
            RandomForestClassifier()
            # GaussianNB(),
            # MultinomialNB(),
            # BernoulliNB(),
            # LogisticRegression()
        ]

        for alg in algs:
            alg = alg.fit(X_train, y_train)
            print(f"{type(alg).__name__}: {alg.score(X_test, y_test)}", file=f)

        print("", file=f)
        print("", file=f)
