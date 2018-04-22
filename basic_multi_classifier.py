import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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


#### USING OTHER DATA SET ####
df_6 = pd.read_csv('training_data/text_emotion.csv')
df_6 = df_6[['sentiment', 'content']]
emotion_new_list = df_6.sentiment.unique()
counts = df_6['sentiment'].value_counts()
# Only use top 6 emotions
keep_cols = ['neutral', 'worry', 'happiness', 'sadness', 'love', 'surprise']
drop_cols = ['anger', 'boredom', 'enthusiasm', 'empty', 'hate', 'relief', 'fun']
df_6 = df_6[df_6['sentiment'].isin(keep_cols)]
dummy_df_6 = pd.get_dummies(df_6['sentiment'])
df_6 = pd.concat([df_6, dummy_df_6], axis=1)
df_6.drop(df_6.columns[0], axis=1, inplace=True)

df_6['content'] = df_6['content'].apply(tokenize)
emotions = df_6.columns[1:]
tweets = df_6['content'].values
labels = df_6[emotions].values
fill = 12
##############################
# train_df = pd.read_csv('training_data/2018-E-c-En-train.txt',
#                        sep='\t',
#                        quoting=3,
#                        lineterminator='\r')
#
# val_df = pd.read_csv('training_data/2018-E-c-En-dev.txt',
#                      sep='\t',
#                      quoting=3,
#                      lineterminator='\r')
#
# df = train_df.append(val_df, ignore_index=True)
#
# # Clean up training data
# df.dropna(axis=0, inplace=True)
# df.reset_index(drop=True, inplace=True)
# df['Tweet'] = df['Tweet'].apply(tokenize)
# emotions = df.columns[2:]
#
# tweets = df['Tweet'].values
# labels = df[emotions].values
#
cv = CountVectorizer()
x_tokens = cv.fit_transform(tweets)
print(f"Input shape: {x_tokens.shape}")

x_train, x_val, y_train, y_val = train_test_split(x_tokens, labels,
                                                  test_size=0.4)

# Suitable classifier models (uncomment the one you'd like to use)
clf = MLPClassifier()
# clf = KNeighborsClassifier()
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()

clf.fit(x_train, y_train)

predicted = clf.predict(x_val)

jaccard_sim = jaccard_similarity_score(y_val, predicted)
prec_score_micro = precision_score(y_val, predicted, average='micro')
prec_score_macro = precision_score(y_val, predicted, average='macro')
rec_score_micro = recall_score(y_val, predicted, average='micro')
rec_score_macro = recall_score(y_val, predicted, average='macro')
f1_micro = f1_score(y_val, predicted, average='micro')
f1_macro = f1_score(y_val, predicted, average='macro')
class_report = classification_report(y_val, predicted, target_names=emotions)

print(f"Jaccard Similarity (accuracy): {jaccard_sim}")
print(f"Classification Report: \n{class_report}")
print(f"Precision Score (micro): {prec_score_micro}")
print(f"Precision Score (macro): {prec_score_macro}")
print(f"Recall Score (micro): {rec_score_micro}")
print(f"Recall Score (macro): {rec_score_macro}")
print(f"f1 Score (micro): {f1_micro}")
print(f"f1 Score (macro): {f1_macro}")

with open('results/combined_accuracy.txt', 'a') as f:
    f.write("\nBasic Model: Sklearn MLP (other dataset- multiclass)\n")
    f.write(f"Jaccard Similarity (accuracy): {jaccard_sim}\n")
    f.write(f"Classification Report:\n {class_report}\n")
    f.write(f"Precision Score (micro): {prec_score_micro}\n")
    f.write(f"Precision Score (macro): {prec_score_macro}\n")
    f.write(f"Recall Score (micro): {rec_score_micro}\n")
    f.write(f"Recall Score (macro): {rec_score_macro}\n")
    f.write(f"f1 Score (micro): {f1_micro}\n")
    f.write(f"f1 Score (macro): {f1_macro}\n")
    f.write('\n')

print(f"Accuracy Score: {accuracy_score(y_val, predicted)}")


fill = 12
