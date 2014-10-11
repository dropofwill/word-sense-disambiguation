"""
Word Sense Disambiguation through WordNet Lookup and a Simplified Lesk
resolution step.

TODO:
    Testing
    Are the two terms Hypo/Hypernyms
"""

import sys
import string
import argparse
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def base_form_of(word):
    return wn.morphy(word)

def synsets_of(word):
    return wn.synsets(word)

def pos_of(synset):
    return set(sense.pos for sense in synset)

def wu_p_sim(result1, result2):
    """
    Input: Two results dicts
    Returns: Similarity of the most common senses of two types
    ---
    A thesaurus similarity algorithm based on the depth in the taxonomy and the
    most specific ancestor node that is common to both of them.
    """
    return result1["synsets"][0].wup_similarity(result2["synsets"][0])

def l_ch_sim(result1, result2):
    """
    Input: Two results dicts
    Returns: Similarity of the most common senses of two types
    ---
    A thesaurus similarity algorithm based on the shortest connecting path and
    how deep in the taxonomy they are.
    """
    return result1["synsets"][0].lch_similarity(result2["synsets"][0])

def compute_overlap(signature, context):
    """
    Input: Two sets of types
    Returns: Length of the intersection of the two sets
    """
    return len(signature & context)

def simple_lesk(results, context):
    """
    Input: A previously computed results dict and a context list/set
    Returns: The sense from WordNet with the most overlap with the context,
             if there are none it defaults to the most common sense
    """
    best_sense = results["synsets"][0]
    max_overlap = 0
    stop_words = stopwords.words("english")
    # ignore base form for overlap?
    stop_words.extend([results["base"], results["word"]])

    context = set(w.lower() for w in tokenizer.tokenize(context) \
                    if w.lower() not in stop_words)
    print context

    for sense in results["synsets"]:
        signature = set()
        text = [ex for ex in sense.examples]
        text.append(sense.definition)

        signature = set(w.lower() for w in tokenizer.tokenize(" ".join(text)) \
                            if w.lower() not in stop_words)

        overlap = compute_overlap(signature, context)

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense

def print_results(results):
    print("{:<20}{:>}"
            .format("Word:", results["word"]))

    print("{:<20}{:>}"
            .format("Base Form:", results["base"]))

    # hacky way to print an awkward list neatly
    print("{:<20}{:>}"
            .format("Synsets:", results["synsets"][0]))
    for syn in results["synsets"][1:]:
        print("{:<20}{:>}".format("", syn))

    print("{:<20}{:>}"
            .format("Possible POS tags:", ", ".join(results["pos_tags"])))

    print("{:<20}{:>}"
            .format("Best sense via Lesk:", results["best_sense"]))

def get_results(word_list, word_context):
    list_results = []
    for i, w in enumerate(word_list):
        res = dict()
        res["word"] = w
        res["base"] = base_form_of(w)
        res["synsets"] = synsets_of(res["base"])
        res["pos_tags"] = pos_of(res["synsets"])
        res["best_sense"] = simple_lesk(res, word_context[i])
        print_results(res)
        list_results.append(res)
    print("\n")

    return list_results

def main(args):
    print args
    words, contexts = [], []

    words.append(args.word1 if args.word1 else raw_input("Enter a word to compare >> "))
    words.append(args.word2 if args.word2 else raw_input("Enter a word to compare >> "))

    contexts.append(args.context1 if args.context1 else raw_input("Enter a context for " + words[0] + " >> "))
    contexts.append(args.context2 if args.context2 else raw_input("Enter a context for " + words[1] + " >> "))

    results = get_results(words, contexts)

    print("{:<20}{:>}"
            .format("Leacock-Chodorow Similarity: ",
               l_ch_sim(results[0], results[1])))

    print("{:<20}{:>}"
            .format("Wu-Palmer Similarity: ",
               wu_p_sim(results[0], results[1])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-w1", "-w", "--word1", "--word",
                          help = "pass an ambiguous word" )
    parser.add_argument( "-w2", "--word2",
                          help = "pass a second ambiguous word" )
    parser.add_argument( "-c1", "-c", "--context1", "--context",
                          help = "pass a context for word 1" )
    parser.add_argument( "-c2", "--context2",
                          help = "pass a context for word 2" )
    args = parser.parse_args()

    main(args)
