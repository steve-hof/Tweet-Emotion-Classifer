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

class NNClassifier():
    
    def __init__(self):
        self.max_dimensions = 300
        self.max_tweet_length = 35
        self.batchSize = 24
        self.testBatchSize = 24
        self.lstmUnits = 64 # was 64
        self.numClasses = 2
        self.iterations = 500 # was 100,000
        
        # Get word2vec stuff
        self.wordsList = np.load('training_data/wordsList.npy')
        print('Loaded the word list!')
        self.wordsList = self.wordsList.tolist() #Originally loaded as numpy array
        self.wordsList = [word.decode('UTF-8') for word in self.wordsList] #Encode words as UTF-8
        self.wordVectors = np.load('training_data/wordVectors.npy')
        print ('Loaded the word vectors!')

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
        for i in range(0,n):
            tweet = re.sub('[^a-zA-Z#]', ' ', X[i])  
            tweet = tweet.lower()
            clean_tweets.append(tweet)
        return clean_tweets
    
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

    def trainNet(self, num_has_emotion, num_no_emotion, emotion):
        tf.reset_default_graph()
        
        # Set up placeholders (one for input data and one for labels)
        labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        input_data = tf.placeholder(tf.int32, [self.batchSize, self.max_tweet_length])
        
        # Get vector
        data = tf.Variable(tf.zeros([self.batchSize, self.max_tweet_length, self.max_dimensions]),dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.wordVectors, input_data)
        
        # Set up LSTM cell then wrap cell in dropout layer to avoid overfitting
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        
        # Combine the 3D input with the LSTM cell and set up network
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        # Cross entropy loss with a softmax layer on top
        # Using Adam for optimizer with 0.0001 learning rate
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        # Tensorboard set-up
        tf.summary.scalar('Training Loss', loss)
        tf.summary.scalar('Training Accuracy', accuracy)
        # tf.summary.scalar('Testing Accuracy', test_accuracy)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for i in range(self.iterations):
            #Next Batch of reviews
            nextBatch, nextBatchLabels = self._getTrainBatch(num_has_emotion, num_no_emotion);
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            #Write summary to board
            if (i % 50 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)

            #Save network every so often
            if (i % 100 == 0 and i != 0):
                save_path = saver.save(sess, f"models/{emotion}_pretrained_lstm.ckpt", global_step=i)
                print(f"saved to {save_path}")
        writer.close()

        # testing
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
    
    def forPreTrainNet(self, num_has_emotion, num_no_emotion):
        tf.reset_default_graph()
        
        # Set up placeholders (one for input data and one for labels)
        labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        input_data = tf.placeholder(tf.int32, [self.batchSize, self.max_tweet_length])
        
        # Get vector
        data = tf.Variable(tf.zeros([self.batchSize, self.max_tweet_length, self.max_dimensions]),dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.wordVectors, input_data)
        
        # Set up LSTM cell then wrap cell in dropout layer to avoid overfitting
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        
        # Combine the 3D input with the LSTM cell and set up network
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        # Cross entropy loss with a softmax layer on top
        # Using Adam for optimizer and going with default 0.001 learning rate
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        
        #load trained network
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('models'))

        # testing
        iterations = 20
        counter = 0
        accuracies = []
        for i in range(iterations):
            nextBatch, nextBatchLabels = self._getTestBatch(num_has_emotion, num_no_emotion);
            print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
            accuracies.append(sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}))
        average_accuracy = np.asarray(accuracies).mean()
        print(f"Average Accuracy: {average_accuracy}")

if __name__ == '__main__':
    #######################################
    #### Download and Clean Tweet Data ####
    #######################################
    
    training_file = 'training_data/2018-E-c-En-train.txt'
    dataset = pd.read_csv(training_file, sep = '\t', quoting = 3, 
                        lineterminator = '\r')
    emotions = dataset.columns.tolist()[2:]
    
    ###################################
    #### Initialize Neural Network ####
    ###################################
    
    nn = NNClassifier()
    
    ############################################################
    #### Calcuate Tweet Vector IDs from word2vec Embeddings ####
    ############################################################
    # for emotion in emotions:
    emotion = 'joy'
    num_has_emotion, num_no_emotion, ids = nn.prepareData(dataset, emotion)
    
    
    ###############################################
    #### Load IDs (if already done previously) ####
    ###############################################
    
#    ids = np.load('training_data/idsMatrix.npy')

    #############################################################
    #### Train Neural Network and Test Accuracy on Test Data ####
    #############################################################
    
    nn.trainNet(num_has_emotion, num_no_emotion, emotion)
    
    #####################################################
    #### Use this to run pretrained net on test data ####
    #####################################################    
#    nn.forPreTrainNet(2544, 4294)

   
    
    