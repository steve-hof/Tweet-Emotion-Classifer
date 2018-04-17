import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input, Sequential
from keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, Dropout, LSTM, Activation, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.metrics import categorical_accuracy, categorical_crossentropy

from sklearn.dummy import DummyClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_similarity_score, precision_score, recall_score, \
    f1_score

BASE_PATH = ''
EMBEDDINGS_PATH = os.path.abspath("/Users/stevehof/school/comp/Word_Embedding_Files")
GLOVE_PATH = os.path.join(EMBEDDINGS_PATH, "glove.twitter.27B/glove.twitter.27B.50d.txt")
EMBEDDING_DIMENSION = 50
MAX_TWEET_LENGTH = 25
MAX_VOCAB = 20000
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def avg_tweet_length(frame, column):
    return len(frame[column].str.cat(sep=' ').split()) / len(frame[column])


def build_lstm_nn(embed_layer, y_train):
    tweet_input = Input(shape=(MAX_TWEET_LENGTH,), dtype='int32')
    embedded_tweets = embed_layer(tweet_input)
    m = LSTM(64, activation='relu')(embedded_tweets)
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
    m = Dense(12, activation='sigmoid')(m)
    # m = Dense(28, activation='relu')(m)
    output = Dense(y_train.shape[1])(m)
    basic_model = Model(tweet_input, output)
    basic_model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'],
                        optimizer=Adam(LEARNING_RATE))
    return basic_model


embedding_index = {}
with open(GLOVE_PATH) as f:
    for index, line in enumerate(f):
        elements = line.split()
        word = elements[0]
        coefs = np.asarray(elements[1:], dtype=np.float32)
        embedding_index[word] = coefs
        # if index + 1 == MAX_VOCAB:
        #     break

train_df = pd.read_csv('training_data/2018-E-c-En-train.txt',
                       sep='\t',
                       quoting=3,
                       lineterminator='\r')

val_df = pd.read_csv('training_data/2018-E-c-En-dev.txt',
                     sep='\t',
                     quoting=3,
                     lineterminator='\r')

# Combine data frames
df = train_df.append(val_df, ignore_index=True)

# Clean up training data
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
emotions = train_df.columns[2:]

# Turn tweets into 2D integer tensors
tweets = list(df['Tweet'].values)
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens')

data = pad_sequences(sequences, maxlen=MAX_TWEET_LENGTH)

emotions_dict = dict(enumerate(emotions))
labels = df[emotions].values
print(f'Shape of data tensor: {data.shape}')
print(f'Shape of label tensor: {labels.shape}')

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_RATIO * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

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
model = build_basic_nn(embedding_layer, y_train)
# model = build_lstm_nn(embedding_layer, y_train)

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
check_pointer = ModelCheckpoint(filepath="multi_label_best_weights.hdf5",
                                verbose=0, save_best_only=True)  # save best model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
                    callbacks=[monitor, check_pointer], verbose=2, epochs=10)
print(history.history.keys())
model.load_weights('multi_label_best_weights.hdf5')  # load weights from best model


# Plotting
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.savefig('plots/model_accuracy.eps', format='eps', dpi=1000)

plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# Turn probabilities into predictions and compare with ground truth
predictions = model.predict(x_val)
mask = predictions > .5
predicted_classes = mask.astype(int)

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
dummy_score_freq = clf_freq.score(x_val, y_val)
print(f"Dummy (most frequent predictor) score {dummy_score_freq}")

# Calculate Accuracy
jaccard_sim = jaccard_similarity_score(y_val, predicted_classes)
prec_score_micro = precision_score(y_val, predicted_classes, average='micro')
prec_score_macro = precision_score(y_val, predicted_classes, average='macro')
rec_score_micro = recall_score(y_val, predicted_classes, average='micro')
rec_score_macro = recall_score(y_val, predicted_classes, average='macro')
f1_micro = f1_score(y_val, predicted_classes, average='micro')
f1_macro = f1_score(y_val, predicted_classes, average='macro')
ham_loss = hamming_loss(y_val, predicted_classes)

print(f"Jaccard Similarity (accuracy): {jaccard_sim}")
print(f"Precision Score (micro): {prec_score_micro}")
print(f"Precision Score (macro): {prec_score_macro}")
print(f"Recall Score (micro): {rec_score_micro}")
print(f"Recall Score (macro): {rec_score_macro}")
print(f"f1 Score (micro): {f1_micro}")
print(f"f1 Score (macro): {f1_macro}")
print(f"Hamming Loss: {ham_loss}")

with open('results/combined_accuracy.txt', 'a') as f:
    f.write(f"Jaccard Similarity (accuracy): {jaccard_sim}")
    f.write(f"Precision Score (micro): {prec_score_micro}")
    f.write(f"Precision Score (macro): {prec_score_macro}")
    f.write(f"Recall Score (micro): {rec_score_micro}")
    f.write(f"Recall Score (macro): {rec_score_macro}")
    f.write(f"f1 Score (micro): {f1_micro}")
    f.write(f"f1 Score (macro): {f1_macro}")
    f.write(f"Hamming Loss: {ham_loss}")
    f.write('\n')



fill = 12
