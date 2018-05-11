# Models for the Multi-label Emotion Classification of Tweets

### While much has been accomplished, this is still very much a work in progress

The project report for the semester is the [project_report.pdf file](https://github.com/therealstevehof/Tweet-Emotion-Classifer/blob/master/project_report.pdf)

The purpose of this project was to build machine learning models that could properly classify the emotions present in tweets. In doing so, it was to serve as a learning space for natural language processing, as well as both the theory behind and python implementation of algorithmic classifiers. The classification problem was of an eleven way multi-label structure and proved to be a difficult but rewarding first project in Machine Learning. 

The key files are: 

1. preprocess_tf_binary.py
When I initially built the first neural network, the code I wrote took eons to execute. For that reason I included it in another file (this one). While the code now takes next to no time, I’ve left it in this separate file. This file pre-processes the data to feed into tf_binary_clf.py

2. tf_binary_clf.py
This file runs the tensor flow binary classification (GloVe embeddings) algorithm discussed  in the paper (BCLSTM)

3. skl_binary_clf.py
This file contains all the code for executing the binary classification per emotion for all of the Bag of Words models discussed in the paper.

4. skl_multi_label_BoW_clf.py
This file contains all the code for executing the multi-label classification of the Bag of Words models discussed in the paper

5. keras_multi_label_glove.py
This file contains all of the code for executing the GloVe embedding multi-label classification networks discussed in the paper (ie keras-LSTM, Feed Forward Neural Network, etc)


