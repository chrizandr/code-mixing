# -*- coding: utf-8 -*-
# !/usr/bin/python2.7

"""Devanagiri to roman text convertor."""

from data_handler import read_data, write_data, printProgressBar
import sys
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME = "/home/chrizandr/code-mixing/resources/indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES = "/home/chrizandr/code-mixing/resources/indic_nlp_resources"

sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp import loader
loader.load()

from indicnlp.transliterate.unicode_transliterate import ItransTransliterator


def transliterate(data, lang):
    """Transliterator."""
    total = len(data)
    new_data = list()
    for i in range(len(data)):
        printProgressBar(i+1, total, prefix='Progress:', suffix='Complete', length=50)
        new_data.append(ItransTransliterator.to_itrans(data[i], LANG))

    return new_data


if __name__ == "__main__":
    LANG = 'hi'
    INPUT_FILE = "/home/chrizandr/code-mixing/data/IITB.en-hi.hi"
    OUTPUT_FILE = "/home/chrizandr/code-mixing/data/IITB.en-hi.hi.roman"
    print("Reading data")
    original_text = read_data(INPUT_FILE, encoding="UNI")
    print("Transliterating")
    romanized_text = transliterate(original_text, LANG)
    print("Writing to file")
    write_data(OUTPUT_FILE, romanized_text, encoding="UNI")
 
