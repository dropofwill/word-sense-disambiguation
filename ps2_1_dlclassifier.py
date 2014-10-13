"""
Supervised Decision List Classifier for Word Sense Disambiguation.

Using Sklearns Base classifier as a guide for building the API

TODO:
    ---
    Procedure:
    Read Supervised Paper
    Review Class Notes
    Preprocess the data, tokenize, remove punctuation, POS tag, & remove xml bits
    Select features, +/- k word, +/- k POS, k = {1,2} words within +/- 5
    For each feature calculate the log likelihood based on the training corpus
    Smooth the counts, with Laplace/Lidstone for instance
    Rank the rules in a decision list based on probabilities
    Classify test data with the most predictive rule that matches
    ---
    Report:
    1-2 paragraph summary of the paper: assumptions, other use cases?
    Conceptually explain the process
    Report the top-10 rules per case
    Report on the performance:
        For each case what was the test set baseline:
            Prior Probability of the majority class
            Actual Majority class label
        Comparison to baseline, actual accuracy and % error reduction over baseline
        Compute precision and recall by sense on each test set
        Compute macro average by weighing each class equally
        Include confusion matrices for both cases with observations
        Include 3 examples of correct and incorrect classification for each case
    Based on these results provide a discussion/reflection.
        e.g. What influenced the results of the performance metrics?
        What does the confusion matrices tell you?
        What went wrong on the examples?
        What concerns if any did you encounter along the way?
    ---
    Bonus (Up to 20):
    +5 Implement method to prune the decision list or interpolate between rules lower in the list
    +7 Read paper and contrast the method with Yarowsky's method
    +3 Apply another smoothing algorithm such as Witten-Bell or Good Turing. Discuss results.
    +5 Yarowsky is known for the unsupervised version of the decision list. Read the paper and analyze how it differs from the supervised method.
    +7.5 Update code to account for the unsupervised approach
    ---
    Problem 6: 1-2 Creative visualization of the WSD problem. Demonstrate ideas visually.
"""

import sys
import argparse
import pprint
import string
import re
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LaplaceProbDist
from nltk.probability import LidstoneProbDist
from nltk.probability import UniformProbDist
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = PunktWordTokenizer()
#tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
#tokenizer = RegexpTokenizer(r'\S+')

pp = pprint.PrettyPrinter(indent=4)

test_data = \
"""
bass:	Stephan Weidner, the composer and bass player for Boehse Onkelz, a
bass:	valued at $250,000. Another double bass trapped in the room is
*bass:	portion of shrimp, mussels, sea bass and whatnot in a spicy,
*bass:	-- OPTIONAL MATERIAL FOLLOWS.) Striped bass are also being spotted in
*bass:	source of entertainment is pay-per-view bass fishing. Yet this is still
*bass:	herring and the enormous striped bass that feed on them. It
*bass:	and dining on Chilean sea bass and poached peaches. <DOC id="APW20010228.0028"
*bass:	<DOC id="NYT20010802.0256" type="story" > Japan's bass fisherman become homeland heroes New
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
"""

def process_corpus(text):
    # split the text into its individual senses and contexts
    corpus = text.split("\n")
    # split the sense from the context
    corpus = [l.split("\t") for l in corpus if l != '']
    # strip the colon from the sense
    corpus = [[l[0][:-1], l[1]] for l in corpus]
    # remove XML tags from corpus
    corpus = [[l[0], re.sub(r'\<.*?(\>|$)', '', l[1])] for l in corpus]
    # Punkt tokenize the context
    corpus = [[l[0], tokenizer.tokenize(l[1].lower())] for l in corpus]
    # Get rid of stop words and punctuation from the context
    stop_words = stopwords.words("english")
    stop_words.extend(string.punctuation)
    # only keep context words that aren't in our stop words list
    corpus = [[l[0], [w for w in l[1] if w not in stop_words]] for l in corpus]

    pp.pprint(corpus)
    print(stop_words)
    return corpus

def main(args):
    print args
    print test_data
    corpus = process_corpus(test_data)

    prev_word_cfdist = ConditionalFreqDist()
    for l in corpus:
        root = re.sub(r'\*', '', l[0])
        probs(prev_word_cfdist, root, l[0], l[1])
    #prev_word_cpdist = ConditionalProbDist(prev_word_cfdist, LaplaceProbDist)
    #prev_word_cpdist = ConditionalProbDist(prev_word_cfdist, LidstoneProbDist, 0.1)
    prev_word_cpdist = ConditionalProbDist(prev_word_cfdist, UniformProbDist)
    print(prev_word_cpdist["bass"].prob("sea"))

def probs(cfd, root_word, sense_word, context):
    root_word_i = context.index(root_word)
    prev_word_i = root_word_i - 1
    prev_word = context[prev_word_i]

    cfd[sense_word][prev_word] += 1
    #condition = sense_word + "_pword_" + prev_word
    #bigrams = nltk.bigrams(text)
    #cfd_bigrams = Counter(bigrams)
    #cfd_unigrams = Counter(list(text))

    print(root_word, root_word_i)
    print(prev_word, prev_word_i)
    print(context)

class DecisionListClassifier():
    """

    """

    def __init__(self):
        pass

    def fit(data, targets=None):
        pass

    def fit_transform(data):
        X = transform(data)
        fit(X)

    def score(data):
        pass

    def predict(data):
        pass

    def predict_proba(data):
        pass

    def transform(data):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-X", "-t", "--train",
                          help = "pass a file path to the train data" )
    parser.add_argument( "-y", "-s", "--test",
                          help = "pass a file path to the test data" )
    parser.add_argument( "-p", "--predict",
                          help = "pass a phrase to predict the sense of" )
    args = parser.parse_args()
    main(args)
