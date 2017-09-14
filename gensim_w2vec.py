"""Word2Vec using gensim."""

import gensim
import logging


class Sentences(object):
    """Sentence iterator class."""

    def __init__(self, filename):
        """Constructor."""
        self.filename = filename

    def __iter__(self):
        """Iterator."""
        for line in open(self.filename):
            if not line.isspace():
                yield line.lower().split()


def main(INPUT_FILE, OUTPUT_MODEL, OUTPUT_FILE):
    """Main."""
    print("Training model")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = Sentences(INPUT_FILE)
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save(OUTPUT_MODEL)

    gen_index_file(filepath=OUTPUT_FILE, model=OUTPUT_MODEL)


def gen_index_file(filepath, model):
    """Create an index file using the gensim model."""
    model = gensim.models.Word2Vec.load(model)
    model.wv.save_word2vec_format(filepath)


if __name__ == "__main__":
    INPUT_FILE = "data/IITB.en-hi.hi.syll"
    OUTPUT_FILE = "data/parallel.hi.syll"
    OUTPUT_MODEL = "models/parallel.hi.syll.mdl"
    main(INPUT_FILE, OUTPUT_MODEL, OUTPUT_FILE)
