"""
Word Sense Disambiguation through WordNet Lookup and a Simplified Lesk
resolution step.

"""

import sys
import argparse
import nltk
from nltk.corpus import wordnet as wn

def base_form_of(word):
    return wn.morphy(word)

def synsets_of(word):
    return wn.synsets(word)

def pos_of(synset):
    return set(sense.pos for sense in syns)

def simple_lesk(word, context):
    pass

def compute_overlap(signature, context):
    pass

def print_results(results):
    print("{:<20}{:>}"
        .format("Word", results["word"]))

    print("{:<20}{:>}"
        .format("Base Form", results["base"]))

    print("{:<20}{:>}"
        .format("Synsets", results["synsets"]))

    print("{:<20}{:>}"
        .format("Possible POS tags", results["pos_tags"]))

def get_results(word_list, word_context=None):
    for w in word_list:
        res = dict()
        res["word"] = w
        res["base"] = base_form_of(w)
        res["synsets"] = synsets_of(res["base"])
        res["pos_tags"] = synsets_of(res["synsets"])
        print_results(res)

def main(args):
    get_results(["dog"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-w1", "-w", "--word1", "--word",
                          help = "pass an ambiguous word" )
    parser.add_argument( "-w2", "--word2",
                          help = "pass a second ambiguous word" )
    parser.add_argument( "-c1", "-c", "--context", "--context1",
                          help = "pass a context for word 1" )
    parser.add_argument( "-c2", "--context2",
                          help = "pass a context for word 2" )
    args = parser.parse_args()

    main(args)
