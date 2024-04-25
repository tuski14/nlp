# pos tagging using hmm

import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

#import nltk
nltk.download('treebank')
# Load the Penn Treebank dataset
corpus = nltk.corpus.treebank.tagged_sents()

# Split the dataset into training and test sets
train_data = corpus[:4000]
test_data = corpus[3000:]

# Train an HMM POS tagger
hmm_tagger = nltk.hmm.HiddenMarkovModelTrainer().train_supervised(train_data)

# Evaluate the tagger on the test data
test_accuracy = hmm_tagger.evaluate(test_data)

print(f"Test accuracy: {test_accuracy:.2f}")

def pos_tag(sentence, tagger):
    tokens = nltk.tokenize.word_tokenize(sentence)
    tagged = tagger.tag(tokens)
    return tagged
user_input = input("Enter the sentence to tag: ")
print(pos_tag(user_input, hmm_tagger))
