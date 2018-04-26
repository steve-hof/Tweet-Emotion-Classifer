#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, \
    Dropout, LSTM, Activation, Bidirectional, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.metrics import categorical_accuracy, categorical_crossentropy

from sklearn.dummy import DummyClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_similarity_score, \
    classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

BASE_PATH = ''
EMBEDDINGS_PATH = os.path.abspath("/Users/stevehof/school/comp/Word_Embedding_Files")
GLOVE_PATH = os.path.join(EMBEDDINGS_PATH, "glove.twitter.27B/glove.twitter.27B.50d.txt")
EMBEDDING_DIMENSION = 50
MAX_TWEET_LENGTH = 25
MAX_VOCAB = 20000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
FLAGS = re.MULTILINE | re.DOTALL


def fix_split(pattern, string):
    splits = list((m.start(), m.end()) for m in re.finditer(pattern, string))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits] + [len(string)]
    return [string[start:end] for start, end in zip(starts, ends)]


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + fix_split(r"(?=[A-Z])", hashtag_body))  # , flags=FLAGS))
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    # Different regex parts for smiley faces
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
    text = re_sub(r"/", " / ")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def avg_tweet_length(frame, column):
    return len(frame[column].str.cat(sep=' ').split()) / len(frame[column])


def build_bi_directional_lstm_nn(embed_layer, y_train):
    tweet_input = Input(shape=(MAX_TWEET_LENGTH,), dtype='int32')
    embedded_tweets = embed_layer(tweet_input)
    m = Bidirectional(LSTM(24, return_sequences=True))(embedded_tweets)
    m = Bidirectional(LSTM(24))(m)
    output = Dense(y_train.shape[1], activation='sigmoid')(m)
    bi_model = Model(tweet_input, output)
    bi_model.compile(loss='binary_crossentropy',
                     optimizer=Adam(LEARNING_RATE),
                     metrics=['categorical_accuracy'])
    return bi_model


def build_lstm_nn(embed_layer, y_train):
    tweet_input = Input(shape=(MAX_TWEET_LENGTH,), dtype='int32')
    embedded_tweets = embed_layer(tweet_input)
    m = LSTM(24, activation='relu', return_sequences=True, dropout=0.6, recurrent_dropout=0.0)(embedded_tweets)
    m = LSTM(12, activation='relu', dropout=0.4)(m)
    output = Dense(y_train.shape[1], activation='sigmoid')(m)
    lstm_model = Model(tweet_input, output)
    lstm_model.compile(loss='binary_crossentropy',
                       optimizer=Adam(LEARNING_RATE),
                       metrics=['categorical_accuracy'])
    return lstm_model


def build_basic_nn(embed_layer, y_train):
    tweet_input = Input(shape=(MAX_TWEET_LENGTH,), dtype='int32')

    embedded_tweets = embed_layer(tweet_input)
    m = GlobalMaxPooling1D()(embedded_tweets)
    m = Dense(24, activation='relu')(m)
    m = Dense(12, activation='relu')(m)
    output = Dense(y_train.shape[1], activation='sigmoid')(m)

    basic_model = Model(tweet_input, output)
    basic_model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'],
                        optimizer=Adam(LEARNING_RATE))
    return basic_model


def add_no_emo(frame, emos):
    frame['no_emotion'] = frame[emos].sum(axis=1)
    frame['no_emotion'] = np.where(frame['no_emotion'] == 0, 1, 0)
    return frame


def prepare_tweet_data(frame, input):
    frame.dropna(axis=0, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    frame[input] = frame[input].apply(tokenize)
    emotions = frame.columns[2:]
    labels = frame[emotions].values
    tweets = list(frame[input].values)
    return tweets, labels


def main():
    embedding_index = {}
    with open(GLOVE_PATH) as f:
        for index, line in enumerate(f):
            elements = line.split()
            word = elements[0]
            coefs = np.asarray(elements[1:], dtype=np.float32)
            embedding_index[word] = coefs
            if index + 1 == 900:
                break

    # Load Training and Validation Data, then combine frames to shuffle
    train_df = pd.read_csv('training_data/2018-E-c-En-train.txt',
                           sep='\t',
                           quoting=3,
                           lineterminator='\r')


    viewing = train_df.append(train_df.sum(numeric_only=True), ignore_index=True)

    val_df = pd.read_csv('training_data/2018-E-c-En-dev.txt',
                         sep='\t',
                         quoting=3,
                         lineterminator='\r')

    test_df = pd.read_csv('training_data/2018-E-c-En-test.txt',
                         sep='\t',
                         quoting=3,
                         lineterminator='\r')

    df = train_df.append(val_df, ignore_index=True)

    # Clean up data
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Tweet'] = df['Tweet'].apply(tokenize)
    emotions = train_df.columns[2:]
    # df = add_no_emo(df, emotions)
    # emotions = df.columns[2:]

    # Turn tweets into 2D integer tensors
    tweets = list(df['Tweet'].values)
    tokenizer = Tokenizer(num_words=MAX_VOCAB, filters='!"$%&()*+,-./:;?@[\]^_`{|}~')
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens')

    data = pad_sequences(sequences, maxlen=MAX_TWEET_LENGTH)

    # emotions_dict = dict(enumerate(emotions))
    labels = df[emotions].values
    print(f'Shape of data tensor: {data.shape}')
    print(f'Shape of label tensor: {labels.shape}')

    # split the data into a training set and a validation set
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.3)
    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]
    # labels = labels[indices]
    # num_validation_samples = int(VALIDATION_RATIO * data.shape[0])
    #
    # x_train = data[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]
    # x_val = data[-num_validation_samples:]
    # y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix....')

    # prepare embedding matrix
    num_words = min(MAX_VOCAB, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION))
    for word, i in word_index.items():
        if i >= MAX_VOCAB:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIMENSION,
                                weights=[embedding_matrix],
                                input_length=MAX_TWEET_LENGTH,
                                trainable=False)

    print('Training model...')
    # Choose model
    # model = build_basic_nn(embedding_layer, y_train)
    model = build_lstm_nn(embedding_layer, y_train)
    # model = build_bi_directional_lstm_nn(embedding_layer, y_train)

    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=75, verbose=1, mode='min')
    #
    # check_pointer = ModelCheckpoint(filepath="multi_label_best_weights.hdf5",
    #                                 verbose=0, save_best_only=True)  # save best model
    #
    # history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
    #                     callbacks=[monitor, check_pointer], verbose=2, epochs=200)
    #
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
                        verbose=2, epochs=100)
    # print(history.history.keys())
    # model.load_weights('multi_label_best_weights.hdf5')  # load weights from best model

    # Plotting
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('LSTM Categorical Accuracy per Epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_acc_file = 'plots/model_accuracy' + timestamp + '.eps'
    plt.savefig(plot_acc_file, format='eps', dpi=1000)

    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Feed Forward Categorical Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plot_loss_file = 'plots/model_loss' + timestamp + '.eps'
    plt.savefig(plot_loss_file, format='eps', dpi=1000)
    plt.show()

    # Turn probabilities into predictions and compare with ground truth
    predictions = model.predict(x_val)
    mask = predictions > .2
    predicted_classes = mask.astype(int)
    print(f"Predicted Results:")
    print(f"{predicted_classes}")

    print()
    print()
    print('########### RESULTS ###########')

    # Dummy Accuracies
    clf_strat = DummyClassifier(strategy='stratified')
    clf_strat.fit(x_train, y_train)
    dummy_score_strat = clf_strat.score(x_val, y_val)
    print(f"Dummy (random predictor) score: {dummy_score_strat}")

    clf_freq = DummyClassifier(strategy='most_frequent')
    clf_freq.fit(x_train, y_train)
    most_freq = np.zeros_like(predicted_classes)
    print(f"Dummy (most frequent predictor) score {jaccard_similarity_score(y_val, most_freq)}")

    # Calculate Accuracy
    # print(f"\nPredicted Classes: \n {predicted_classes}")
    print("\nActual Accuracies:")
    jaccard_sim = jaccard_similarity_score(y_val, predicted_classes)
    prec_score_micro = precision_score(y_val, predicted_classes, average='micro')
    prec_score_macro = precision_score(y_val, predicted_classes, average='macro')
    rec_score_micro = recall_score(y_val, predicted_classes, average='micro')
    rec_score_macro = recall_score(y_val, predicted_classes, average='macro')
    f1_micro = f1_score(y_val, predicted_classes, average='micro')
    f1_macro = f1_score(y_val, predicted_classes, average='macro')
    ham_loss = hamming_loss(y_val, predicted_classes)
    class_report = classification_report(y_val, predicted_classes, target_names=emotions)

    print(f"Jaccard Similarity (accuracy): {jaccard_sim}")
    print(f"Classification Report: \n{class_report}")
    print(f"Precision Score (micro): {prec_score_micro}")
    print(f"Precision Score (macro): {prec_score_macro}")
    print(f"Recall Score (micro): {rec_score_micro}")
    print(f"Recall Score (macro): {rec_score_macro}")
    print(f"f1 Score (micro): {f1_micro}")
    print(f"f1 Score (macro): {f1_macro}")
    print(f"Hamming Loss: {ham_loss}")

    with open('results/combined_accuracy.txt', 'a') as f:
        f.write(f"\nRun at {timestamp}\n")
        # f.write("Basic Model: 50d embeds, 2 LSTM, 128-relu (d/o=0.6), 12-relu (d/o=0.4), 200 epochs, LR=.001, (cleaned tweets), (bin_c_e), no-no_emo_class, threshold=0.2\n")
        f.write("Feed Forward Model: 50d embeds, 2 Dense, 128-relu, 12-relu, 200 epochs")
        f.write(f"Jaccard Similarity (accuracy): {jaccard_sim}\n")
        f.write(f"Classification Report:\n {class_report}\n")
        f.write(f"Precision Score (micro): {prec_score_micro}\n")
        f.write(f"Precision Score (macro): {prec_score_macro}\n")
        f.write(f"Recall Score (micro): {rec_score_micro}\n")
        f.write(f"Recall Score (macro): {rec_score_macro}\n")
        f.write(f"f1 Score (micro): {f1_micro}\n")
        f.write(f"f1 Score (macro): {f1_macro}\n")
        f.write(f"Hamming Loss: {ham_loss}\n")
        f.write('\n')


    # test_tweets, test_labels = prepare_tweet_data(test_df, 'Tweet')
    # test_sequences = tokenizer.texts_to_sequences(test_tweets)
    # test_data = pad_sequences(test_sequences, maxlen=MAX_TWEET_LENGTH)
    # for i in range(5):
    #     proba = model.predict(test_sequences[i])
    #     test_mask = proba > 0.2
    #     pred = test_mask.astype(int)
    #     print(test_tweets[i])
    #     print(f"Prediction:\n{pred}")
    #     print(f"Truth:\n{test_labels[i]}")



    fill = 12


if __name__ == '__main__':
    main()
