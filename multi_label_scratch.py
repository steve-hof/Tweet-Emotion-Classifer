# Learning Keras multilabel
# Classify toxic Comments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam

plt.style.use('fivethirtyeight')


df = pd.read_csv('training_data/text_emotion.csv')
df = df[['sentiment', 'content']]
emotion_new_list = df.sentiment.unique()
counts = df['sentiment'].value_counts()
keep_cols = ['neutral', 'worry', 'happiness', 'sadness', 'love', 'surprise']
drop_cols = ['anger', 'boredom', 'enthusiasm', 'empty', 'hate', 'relief', 'fun']
df = df[df['sentiment'].isin(keep_cols)]
dummy_df = pd.get_dummies(df['sentiment'])
df = pd.concat([df, dummy_df], axis=1)
df.drop(df.columns[0], axis=1, inplace=True)
fill = 12


mask = np.random.rand(len(df)) < 0.7
train_df = df[mask]
test_df = df[~mask]


X_train = train_df['content'].values
y_train = train_df[['neutral', 'worry', 'happiness', 'sadness', 'love', 'surprise']].values
X_test = test_df['content'].values

max_features = 20000  # number of words we want to keep
maxlen = 35  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 20  # dimension of the hidden variable, i.e. the embedding dimension

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

comment_input = Input((maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen,
                        embeddings_initializer="uniform")(comment_input)

# we add a GlobalMaxPooling1D, which will extract features from the embeddings
# of all words in the comment
h = GlobalMaxPooling1D()(comment_emb)

# We project onto a six-unit output layer, and squash it with a sigmoid:
output = Dense(6, activation='sigmoid')(h)

model = Model(inputs=comment_input, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.01),
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, validation_split=0.1)

fill = 12
