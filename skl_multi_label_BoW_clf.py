#!/usr/bin/env python3

"""
This script preprocesses the tweet data similarily to preprocess_tf_binary.py

It uses the same script for processing hashtags, emojis and usernames
(found at https://gist.github.com/tokestermw/cb87a97113da12acb388)

It then uses sklearn libraries to perform multilabel classification on
the Bag of Words Model algorithms contained below

"""

import pandas as pd
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.metrics import accuracy_score, jaccard_similarity_score, \
    classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier

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


def main():
    train_df = pd.read_csv('training_data/2018-E-c-En-train.txt',
                           sep='\t',
                           quoting=3,
                           lineterminator='\r')

    val_df = pd.read_csv('training_data/2018-E-c-En-dev.txt',
                         sep='\t',
                         quoting=3,
                         lineterminator='\r')

    df = train_df.append(val_df, ignore_index=True)

    # Clean up training data
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Tweet'] = df['Tweet'].apply(tokenize)
    emotions = df.columns[2:]

    # separate into data and labels
    tweets = df['Tweet'].values
    labels = df[emotions].values

    # map tweets to vector representation of the unique words it contains
    cv = CountVectorizer()
    x_tokens = cv.fit_transform(tweets)

    x_train, x_val, y_train, y_val = train_test_split(x_tokens, labels,
                                                      test_size=0.4)

    # suitable classifier models
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)
    # clf = KNeighborsClassifier()
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()

    # fit on the training data
    clf.fit(x_train, y_train)

    # make predictions on testing data
    predicted = clf.predict(x_val)

    # calculate a variety of metrics for comparison
    jaccard_sim = jaccard_similarity_score(y_val, predicted)
    prec_score_micro = precision_score(y_val, predicted, average='micro')
    prec_score_macro = precision_score(y_val, predicted, average='macro')
    rec_score_micro = recall_score(y_val, predicted, average='micro')
    rec_score_macro = recall_score(y_val, predicted, average='macro')
    f1_micro = f1_score(y_val, predicted, average='micro')
    f1_macro = f1_score(y_val, predicted, average='macro')
    class_report = classification_report(y_val, predicted, target_names=emotions)

    # print metrics to terminal
    print(f"Jaccard Similarity (accuracy): {jaccard_sim}")
    print(f"Classification Report: \n{class_report}")
    print(f"Precision Score (micro): {prec_score_micro}")
    print(f"Precision Score (macro): {prec_score_macro}")
    print(f"Recall Score (micro): {rec_score_micro}")
    print(f"Recall Score (macro): {rec_score_macro}")
    print(f"f1 Score (micro): {f1_micro}")
    print(f"f1 Score (macro): {f1_macro}")


if __name__ == '__main__':
    main()
