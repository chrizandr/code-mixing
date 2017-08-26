"""Handle the data."""
import re
import json
import pdb
from ortho import ortho_syllable


def construct_sentence(sentence):
    """Remove unicode and usernames, keep only words."""
    regex = r'([a-zA-Z\'\-]*)\\HI|([a-zA-Z\'\-]*)\\EN'
    try:
        searchObj = re.findall(regex, sentence)
    except:
        pdb.set_trace()
    SW = ""
    for matching in searchObj:
        if len(matching[0]) > 0:
            SW = SW + matching[0] + " "
        elif len(matching[1]) > 0:
            SW = SW + matching[1] + " "
    return SW


def read_data_json(filepath):
    """Read data into a python dictionary."""
    f = open(filepath, 'r')
    data = json.load(f)
    return data


def read_data_tsv(filepath):
    """Read data from a .tsv file."""
    data = list()
    with open(filepath, 'r') as f:
        for line in f:
            try:
                text = line.split('\t')[1]
            except:
                continue
            data.append(text)
    return data


def write_data(filepath, data):
    """Write the data into the file."""
    string = "\n".join(data)
    f = open(filepath, 'w')
    f.write(string)
    f.close()


def get_sentences(data):
    """Get the constructed sentence from the data."""
    sentences = list()
    for obj in data:
        lang_text = obj["lang_tagged_text"]
        if type(lang_text) is float:
            continue
        # for line in lang_text:
        sentence = construct_sentence(lang_text)
        sentences.append(sentence)
    return sentences


def clean_str(string):
    """Tokenization/string cleaning for all datasets except for SST.

    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def break_in_subword(f, add_word=False):
    """Break the test into sub_words."""
    texts = []
    word_texts = []

    with json.load(open(f, "r")) as data:
        for x in data:
            text = data["text"]
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
            cleaned_text = clean_str(text)
            splitted_text = cleaned_text.split()
            joined_text = []
            word_list = []
            for y in splitted_text:
                if not y.isspace():
                    joined_text.append(ortho_syllable(y.strip()))
                    if add_word:
                        word_list.append(y.strip())
            texts.append(joined_text)
            if add_word:
                word_texts.append(word_list)

    if add_word:
        return texts, word_texts
    else:
        return texts


if __name__ == "__main__":
    data = read_data_tsv("conversations.out")
    sentences = get_sentences(data)
    write_data("final_codemixed_extracted.txt", sentences)
