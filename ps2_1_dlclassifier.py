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
*bass:	portion of shrimp, mussels, sea bass and whatnot in a spicy,
bass:	Stephan Weidner, the composer and bass player for Boehse Onkelz, a
bass:	valued at $250,000. Another double bass trapped in the room is
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
        # root word without and with the prepended *
        self.root = None        # + likelihood denotes this sense
        self.root_star = None   # - likelihood denotes this sense
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
        self.decision_list = []
        # Generates the three above attrs from the test data
        self.fit()

    def fit(self):
        # creates self.cfd and self.cpd
        self.generate_distributions()
        self.generate_decision_list()

    def predict(self, context):
        if type(context) != list:
            context = self.clean_text(context)

        for rule in self.decision_list:
            if self.check_rule(context, rule[0]):
                # + implies root, - implies root_star
                if rule[1] > 0:
                    return self.root
                elif rule[1] < 0:
                    return self.root_star

    def generate_distributions(self):
        self.cfd = ConditionalFreqDist()

        self.prev_word_dist(self.corpus)
        self.next_word_dist(self.corpus)

        #self.cpd = ConditionalProbDist(cfd, LaplaceProbDist)
        self.cpd = ConditionalProbDist(self.cfd, LidstoneProbDist, 0.1)
        #self.cpd = ConditionalProbDist(cfd, UniformProbDist)

    def prev_word_dist(self, corpus):
        for line in corpus:
            sense, context = line[0], line[1]
            prev_word = self.prev_word(context, self.root)
            # create freqdist for each sense per word
            condition = "pword_" + prev_word
            self.cfd[condition][sense] += 1

    def next_word_dist(self, corpus):
        for line in corpus:
            sense, context = line[0], line[1]
            next_word = self.next_word(context, self.root)
            # create freqdist for each sense per word
            condition = "nword_" + next_word
            self.cfd[condition][sense] += 1

    def check_next_word(self, context, word):
        next_word = self.next_word(context, self.root)
        return next_word == word

    def check_prev_word(self, context, word):
        prev_word = self.prev_word(context, self.root)
        return prev_word == word

    def prev_word(self, context, word):
        word_i = context.index(word)
        prev_word_i = word_i - 1
        print(context)
        return context[prev_word_i]

    def next_word(self, context, word):
        word_i = context.index(self.root)
        next_word_i = word_i + 1
        return context[next_word_i]

    def check_rule(self, context, rule):
        rule_type, rule_word = rule.split("_")
        #print(rule_type, rule_word)

        if rule_type == "pword":
            #print("pword")
            return self.check_prev_word(context, rule_word)
        elif rule_type == "nword":
            #print("nword")
            return self.check_next_word(context, rule_word)

    def generate_decision_list(self):
        for rule in self.cpd.conditions():
            likelihood = self.calculate_log_likelihood(rule)
            self.decision_list.append([rule, likelihood])
        # instead of always applying the abs, I opted to apply only while
        # sorting, as the sign is an easy way to denote sense:
        # + for root / - for root star
        self.decision_list.sort(key=lambda rule: math.fabs(rule[1]), reverse=True)
        pp.pprint(self.decision_list)

    def calculate_log_likelihood(self, rule):
        prob = self.cpd[rule].prob(self.root)
        prob_star = self.cpd[rule].prob(self.root_star)
        div = prob / prob_star
        # -means prob_star, +means prob
        return math.log(div, 2)
        #return math.fabs(math.log(div, 2))

    def score(self, data):
        pass

    def process_corpus(self, text):
        """
        Process an input of the form:
        word*   context of word to train the classifier on
        word    context of the word with a different sense
        """
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
        # get root word without the * by looking at the first example
        self.root = re.sub(r'\*', '', corpus[0][0])
        self.root_star = "*" + self.root
        #pp.pprint(corpus)
        return corpus

    def clean_text(self, text):
        """
        Process raw text without sense information
        """
        # remove XML tags from corpus
        text = re.sub(r'\<.*?(\>|$)', '', text)
        # Punkt tokenize the context
        text = tokenizer.tokenize(text)
        # Get rid of stop words and punctuation from the context
        stop_words = stopwords.words("english")
        stop_words.extend(string.punctuation)
        # only keep context words that aren't in our stop words list
        text = [w.lower() for w in text if w not in stop_words]
        #pp.pprint(text)
        return text


    def test_based_on_paper_results(self):
        # Should equal ~7.14 according to Yarowsky given test data string
        print(self.calculate_log_likelihood("pword_sea"))

def main(args):
    #print args
    #print test_data
    #corpus = process_corpus(test_data)
    #fit(corpus)
    clf = DecisionListClf(test_data)
    clf.test_based_on_paper_results()
    print(clf.predict("restaurant, waiters serve pecan-crusted sea bass ($18.95) and peppered rib eye"))

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
