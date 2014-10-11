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

def main(args):
    print args

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
