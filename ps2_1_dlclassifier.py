# -*- coding: utf-8 -*-
"""
Supervised Decision List Classifier for Word Sense Disambiguation.

TODO:
    ---
    Procedure:
    √ Read Supervised Paper
    √ Review Class Notes
    √ Preprocess the data, tokenize, remove punctuation, POS tag, & remove xml bits
    Select features, +/- k word, +/- k POS, k = {1,2} words within +/- 5
    √ For each feature calculate the log likelihood based on the training corpus
    √ Smooth the counts, with Laplace/Lidstone for instance
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
import math
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
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
*bass:	restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye
"""

class DecisionListClf(object):
    """
    Implements the Supervised Yarowsky Decision list for the homograph
    disambiguation problem.

    Constructor takes as input a string of test data of the form:
    word*   context of word to train the classifier on
    word    context of the word with a different sense
    """

    def __init__(self, test_data):
        # The corpus split into a two part list [sense, [context, tokens]]
        # It handles basic normalization by removing English stop words,
        # punctuation, and XML tags
        self.corpus = self.process_corpus(test_data)
        # Uses NLTK's ConditionalFreqDist to count up frequencies
        self.cfd = None
        # Uses NLTK's ConditionalProbDist and one of the ProbDistI classes to
        # calculate probabilities and smooth counts respectively
        self.cpd = None
        # Creates a list of rules sorted by their log likelihood
        self.decision_list = None
        # Generates the three above attrs from the test data
        self.fit(self.corpus)

    def fit(self, corpus):
        # creates self.cfd and self.cpd
        self.generate_distributions(corpus)
        self.decision_list = self.generate_decision_list(self.cpd)

    def generate_distributions(self, corpus):
        self.cfd = ConditionalFreqDist()

        self.get_prev_word_dist(corpus)
        self.get_next_word_dist(corpus)

        #self.cpd = ConditionalProbDist(cfd, LaplaceProbDist)
        self.cpd = ConditionalProbDist(self.cfd, LidstoneProbDist, 0.1)
        #self.cpd = ConditionalProbDist(cfd, UniformProbDist)

    def get_prev_word_dist(self, corpus):
        for line in corpus:
            sense = line[0]
            context = line[1]
            # remove the * marking the sense
            root_word = re.sub(r'\*', '', line[0])
            root_word_i = context.index(root_word)
            prev_word_i = root_word_i - 1
            prev_word = context[prev_word_i]
            # create freqdist for each sense per word
            condition = "pword_" + prev_word
            self.cfd[condition][sense] += 1

    def get_next_word_dist(self, corpus):
        for line in corpus:
            sense = line[0]
            context = line[1]
            # remove the * marking the sense
            root_word = re.sub(r'\*', '', line[0])
            root_word_i = context.index(root_word)
            next_word_i = root_word_i + 1
            next_word = context[next_word_i]
            # create freqdist for each sense per word
            condition = "nword_" + next_word
            self.cfd[condition][sense] += 1

    def generate_decision_list(self, cpd):
        pass

    def score(self, data):
        pass

    def predict(self, data):
        pass

    def process_corpus(self, text):
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
        #pp.pprint(corpus)
        #print(stop_words)
        return corpus

    def test_based_on_paper_results(self):
        # Should equal 7.14 according to Yarowsky given test data string
        sea_fish = self.cpd["pword_sea"].prob('bass')
        sea_music = self.cpd["pword_sea"].prob('*bass')
        sea_div = sea_fish / sea_music
        sea_log = math.log(sea_div, 2)
        sea_abs = math.fabs(sea_log)

        print(self.cpd.conditions())
        print(sea_fish)
        print(sea_music)
        print(sea_div)
        print(sea_log)
        print(sea_abs)


def main(args):
    #print args
    #print test_data
    #corpus = process_corpus(test_data)
    #fit(corpus)
    clf = DecisionListClf(test_data)
    clf.test_based_on_paper_results()

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
