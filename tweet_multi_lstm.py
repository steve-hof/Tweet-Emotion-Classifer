#!/usr/bin/env python3

import numpy as np
import os
import pickle
import tensorflow as tf
import datetime
from random import randint

EMBEDDING_DIMENSION = 50
MAX_TWEET_LENGTH = 35
BATCH_SIZE = 24
LSTM_UNITS = 32
NUM_CLASSES = 2
ITERATIONS = 8000
LEARNING_RATE = 1e-3
NUM_HIDDEN = 2
DROPOUT_KEEP_PROB = .55
# NUM_STEPS = 35

hyp_str = "nLST-" + str(LSTM_UNITS) + "lr-" + str(LEARNING_RATE) + \
          "n_hid-" + str(NUM_HIDDEN) + \
          "d_Prob-" + str(DROPOUT_KEEP_PROB) + \
          "bSize-" + str(BATCH_SIZE) + "_bal-T_"


def _get_train_batch(train_emo_ids, train_no_emo_ids):
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_TWEET_LENGTH])
    for i in range(BATCH_SIZE):
        if i % 2 == 0:
            num = randint(1, len(train_emo_ids))
            arr[i] = train_emo_ids[num - 1:num]
            labels.append([1, 0])
        else:
            num = randint(1, len(train_no_emo_ids))
            arr[i] = train_no_emo_ids[num - 1:num]
            labels.append([0, 1])
    return arr, labels


def _get_test_batch(test_emo_ids, test_no_emo_ids):
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_TWEET_LENGTH])
    for i in range(BATCH_SIZE):
        if i % 2 == 0:
            num = randint(1, len(test_emo_ids))
            arr[i] = test_emo_ids[num - 1:num]
            labels.append([1, 0])
        else:
            num = randint(1, len(test_no_emo_ids))
            arr[i] = test_no_emo_ids[num - 1:num]
            labels.append([0, 1])
    return arr, labels


def main():
    # load up the saved ids and weights
    pickle_path = 'pre_processed_pickles/anger/balanced_tweet_data_' + \
                  str(EMBEDDING_DIMENSION) + 'd.pickle'
    with open(pickle_path, 'rb') as f:
        train_emo, train_no_emo, test_emo, test_no_emo, weights = pickle.load(f)

    # BUILD THE ACTUAL NN

    tf.reset_default_graph()

    # Set up placeholders for input and labels
    with tf.name_scope("Labels") as scope:
        labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
    with tf.name_scope("Input") as scope:
        input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_TWEET_LENGTH])

    # Get embedding vector
    with tf.name_scope('Embeds_Layer') as scope:
        embedding = tf.Variable(tf.zeros([len(weights), EMBEDDING_DIMENSION]), dtype=tf.float32,
                                name='embedding')
        embed = tf.nn.embedding_lookup(embedding, input_data)  # maybe change 'embedding back to 'weights'
        embed = tf.transpose(embed, [1, 0, 2], name='last_lstm')
        embed = tf.gather(embed, int(embed.get_shape()[0]) - 1)
        # embed = tf.unstack(embed, axis=1, name='unpck')

    # Set up LSTM cell then wrap cell in dropout layer to avoid over fitting
    with tf.name_scope("LSTM_Cell") as scope:
        lstm_fw_cell = tf.contrib.rnn.BasicRNNCell(LSTM_UNITS)
        lstm_bw_cell = tf.contrib.rnn.BasicRNNCell(LSTM_UNITS)
        rnn_cell = tf.contrib.rnn.BasicRNNCell(LSTM_UNITS)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=DROPOUT_KEEP_PROB)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([rnn_cell for _ in range(NUM_HIDDEN)], state_is_tuple=True)
        initial_state = stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

    with tf.name_scope("RNN_Forward") as scope:
        value, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, embed)

    with tf.name_scope("Fully_Connected") as scope:
        weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='bias')
        value = tf.transpose(value, [1, 0, 2], name='last_lstm')
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        tf.summary.histogram("weights", weight)
        tf.summary.histogram("biases", bias)

    with tf.name_scope("Predictions") as scope:
        prediction = (tf.matmul(last, weight) + bias)

    # Cross entropy loss with a softmax layer on top
    # Using Adam for optimizer
    with tf.name_scope("Loss_and_Accuracy") as scope:
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

    with tf.name_scope("Training") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Output directory for models and summaries
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = hyp_str + timestamp
    logdir = os.path.abspath(os.path.join(os.path.curdir, "temp_50d/tboard", name))
    print(f"Writing to {logdir}")

    # Summaries for loss and accuracy
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    loss_summary = tf.summary.scalar('Loss', loss)

    # Training summaries
    train_summary_dir = os.path.join(logdir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Testing summaries
    test_summary_dir = os.path.join(logdir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    summary_op = tf.summary.merge_all()

    # Checkpointing
    checkpoint_dir = os.path.abspath(os.path.join(logdir, "checkpoints"))

    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for i in range(ITERATIONS):
        # Next Batch of blurbs
        next_train_batch, next_train_batch_labels = _get_train_batch(train_emo, train_no_emo);
        sess.run(optimizer, {input_data: next_train_batch, labels: next_train_batch_labels})

        # Write training summary to board
        if i % 100 == 0:
            summary = sess.run(summary_op, {input_data: next_train_batch, labels: next_train_batch_labels})
            train_summary_writer.add_summary(summary, i)
            train_summary_writer.flush()

            next_test_batch, next_test_batch_labels = _get_test_batch(test_emo, test_no_emo);
            testSummary = sess.run(summary_op, {input_data: next_test_batch, labels: next_test_batch_labels})
            test_summary_writer.add_summary(testSummary, i)
            test_summary_writer.flush()

        # Save network every so often
        if i % 1000 == 0 and i != 0:
            save_path = saver.save(sess, f"temp_50d/models/_pretrained_lstm.ckpt", global_step=i)
            print(f"saved to {save_path}")

    train_summary_writer.close()
    test_summary_writer.close()


if __name__ == '__main__':
    main()
