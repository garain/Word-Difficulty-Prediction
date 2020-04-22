import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


def load_data():
    data = pd.read_csv('I159729.csv')
    data = data.dropna()
    data.drop(data.columns[[1, 2, 3, 4, 6, 7, 8]], axis=1, inplace=True)
    data["Word"] = data["Word"].str.lower().replace("'", "")

    data["Difficulty"] = 0
    data.loc[data["I_Zscore"] > -0.3, ["Difficulty"]] = 1
    data.loc[data["I_Zscore"] > 0.3, ["Difficulty"]] = 2
    # data.loc[data["I_Zscore"] > 0.3, ["Difficulty"]] = 3
    
    data_x = data["Word"]
    data_x = np.array(data_x)
    data_y = data["Difficulty"]
    data_y = np.array(data_y)
    data_y = to_categorical(data_y)
    
    return data_x, data_y


def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():

    alphabet = set(list(string.ascii_lowercase))
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet
    
'''
data_x, data_y = load_data()
print(data_x)
print(data_y)
# Max len -> 21

vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()
print(vocab)

input_data = encode_data(data_x, 21, vocab)
print(input_data)
'''
