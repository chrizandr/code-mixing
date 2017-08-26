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
            yield line.split()


def main():
    """Main."""
    print("Training model")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = Sentences("final_codemixed_extracted.txt")
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save("english_w2vec_gensim.mdl")

    gen_index_file(filepath="english_w2vec.txt", model="english_w2vec_gensim.mdl")


def gen_index_file(filepath, model):
    """Create an index file using the gensim model."""
    model = gensim.models.Word2Vec.load(model)
    model.wv.save_word2vec_format(filepath)


if __name__ == "__main__":
    main()
