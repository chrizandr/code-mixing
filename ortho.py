"""Orthagraphic syllable splitting."""

import numpy as np


def ortho_syllable(word):
    """Split word to orhtographic syllable."""
    vector = vectorize(word)
    grad_vec = gradient(vector)
    SW = ""
    i = 0
    w_len = len(word)
    while(i < w_len):
        SW = SW + word[i]
        if (i+1) < w_len:
            if i == 0 and grad_vec[i] == -1:
                SW = SW + word[i+1] + " "
                i += 1
            elif grad_vec[i] == -1 and i != w_len-1:
                if word[i+1] in ['r', 's', 't', 'l', 'n', 'd'] and i+1 != w_len-1:
                    if vector[i+2] == 0:
                        SW = SW + word[i+1]
                        i += 1
                SW = SW + " "
        i += 1
    # pdb.set_trace()
    return SW.split()


def is_vowel(char):
    """Check if it is vowel."""
    return char in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']


def gradient(vector):
    """Get the gradient of the vector."""
    vec2 = vector[1::]
    vec2.append(0)
    vec2 = np.array(vec2)
    vec = np.array(vector)
    return vec2-vec


def vectorize(word):
    """Vectorize based on consonant and vowel."""
    vec = list()
    for i in range(len(word)):
        vec.append(int(is_vowel(word[i])))
    return vec


if __name__ == "__main__":
    vec = ortho_syllable("prarthna")
    print(vec)
