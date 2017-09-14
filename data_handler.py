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


def read_data(filename, encoding="ASCII"):
    """Read sentences from the file."""
    f = open(filename, "r")
    data = list()
    for line in f:
        if encoding == "UNI":
            data.append(line.decode('utf-8'))
        else:
            data.append(line)

    return data


def read_data_json(filepath):
    """Read data into a python dictionary."""
    f = open(filepath, 'r')
    data = json.load(f)
    return data


def text_from_json(f):
    """Get sentences from the JSON file."""
    f = open(f, 'r')
    data = json.load(f)
    texts = [x["text"] for x in data]
    return texts


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


def write_data(filepath, data, encoding="ASCII"):
    """Write the data into the file."""
    if encoding == "UNI":
        string = "\n".join([x.encode("ascii", "ignore") for x in data])
    else:
        string = "\n".join(data)
    with open(filepath, 'w') as f:
        f.write(string)
    return None


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


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='>'):
    """
    Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


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
    string = re.sub(r"[.,#!$%&;:{}=_`~()/\\]", "", string)
    return string.strip().lower()


def break_in_subword(data, add_word=False, sentences=False):
    """Break the text into sub_words."""
    texts = []
    word_texts = []
    i = 0
    total = len(data)
    for text in data:
        printProgressBar(i+1, total, prefix='Progress:', suffix='Complete', length=50)
        i += 1
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
        cleaned_text = clean_str(text)
        splitted_text = cleaned_text.split()
        joined_text = []
        word_list = []
        for y in splitted_text:
            if(not y.isspace()):
                if sentences:
                    joined_text.append(" ".join(ortho_syllable(y.strip())))
                else:
                    joined_text.append(ortho_syllable(y.strip()))
                if add_word:
                    word_list.append(y.strip())
        if sentences:
            texts.append(" ".join(joined_text))
        else:
            texts.append(joined_text)
        if add_word:
            word_texts.append(word_list)
    if add_word:
        return texts, word_texts
    else:
        return texts


if __name__ == "__main__":
    print("Reading from file")
    data = read_data_tsv("conversations.out")
    print("Cleaning words")
    broken_words = break_in_subword(data)
    print("Writing words")
    write_data("conversations_cleaned.txt", broken_words)
    # sentences = get_sentences(data)
    # write_data("final_codemixed_extracted.txt", sentences)
