"""Handle the data."""
import re
import json
import pdb


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


def read_data(filepath):
    """Read data into a python dictionary."""
    f = open(filepath, 'r')
    data = json.load(f)
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


if __name__ == "__main__":
    data = read_data("final_codemixed.json")
    sentences = get_sentences(data)
    write_data("final_codemixed_extracted.txt", sentences)
