import re
import string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.metrics import categorical_accuracy, categorical_crossentropy

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

INTENDED_EMBEDDING_DIMENSION = 50
max_vocab = 20000  # number of words we want to keep
max_tweet_length = 35  # max length of the tweets in the model
batch_size = 64  # batch size for the model
embedding_dims = 50  # dimension of the hidden variable, i.e. the embedding dimension
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


# Scatter plot for word embeddings

def plot_words(data, start, stop, step, word_list):
    trace = go.Scatter(
        x=data[start:stop:step, 0],
        y=data[start:stop:step, 1],
        mode='markers',
        text=word_list[start:stop:step]
    )
    layout = dict(title='t-SNE 1 vs t-SNE 2',
                  yaxis=dict(title='t-SNE 2'),
                  xaxis=dict(title='t-SNE 1'),
                  hovermode='closest')
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


def main():
    train_df = pd.read_csv('training_data/2018-E-c-En-train.txt',
                           sep='\t',
                           quoting=3,
                           lineterminator='\r')

    test_df = pd.read_csv('training_data/2018-E-c-En-dev.txt',
                          sep='\t',
                          quoting=3,
                          lineterminator='\r')

    glove_file_path = "/Users/stevehof/school/comp/Word_Embedding_Files/" \
                      "glove.twitter.27B/glove.twitter.27B." + \
                      str(INTENDED_EMBEDDING_DIMENSION) + "d.txt"

    train_df.drop(train_df.columns[0], axis=1, inplace=True)
    test_df.drop(test_df.columns[0], axis=1, inplace=True)

    train_df.dropna(axis=0, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df.dropna(axis=0, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df['Tweet'] = train_df['Tweet'].apply(tokenize)
    test_df['Tweet'] = test_df['Tweet'].apply(tokenize)
    emotions = train_df.columns[1:]

    X_train = train_df['Tweet'].values
    y_train = train_df[emotions].values
    X_test = test_df['Tweet'].values
    y_test = test_df[emotions].values

    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    x_train = tokenizer.texts_to_sequences(X_train)
    x_test = tokenizer.texts_to_sequences(X_test)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    x_train = pad_sequences(x_train, maxlen=max_tweet_length)
    x_test = pad_sequences(x_test, maxlen=max_tweet_length)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    embeddings_index = dict()

    with open(glove_file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((max_vocab, embedding_dims))
    for word, index in tokenizer.word_index.items():
        if index > max_vocab - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    # Create Neural Network
    model_glove = Sequential()
    model_glove.add(Embedding(max_vocab, embedding_dims, input_length=max_tweet_length,
                              weights=[embedding_matrix], trainable=False))
    model_glove.add(Dropout(0.2))
    model_glove.add(LSTM(embedding_dims, activation='relu', return_sequences=True))
    model_glove.add(Dropout(0.2))
    model_glove.add(Conv1D(35, 11, activation='relu'))
    model_glove.add(MaxPooling1D(pool_size=11))
    model_glove.add(LSTM(embedding_dims, activation='relu'))
    model_glove.add(Dense(11, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=True)

    model_glove.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    history = model_glove.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=12)

    # Plotting
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    fill = 12


if __name__ == '__main__':
    main()