""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
from collections import Counter
import re

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def tokenize(text):
    # TODO customize to your needs
    text = text.lower()
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class Features:
    def __init__(self, data_file, num_merges=10000):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))

        self.tokenized_text = [tokenize(text) for text in texts]
        self.labelset = list(set(self.labels))
        
        # BPE
        vocab = Counter(word for text in self.tokenized_text for word in text)
        vocab = {' '.join(word): count for word, count in vocab.items()}
        
        self.bpe_codes = {}
        for i in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            self.bpe_codes[best] = i

        self.vocab = set(vocab.keys())

    def apply_bpe(self, word):
        word = ' '.join(word)
        while True:
            pairs = get_stats({word: 1})
            if not pairs:
                break
            bigram = max(pairs, key=lambda pair: self.bpe_codes.get(pair, float('-inf')))
            if bigram not in self.bpe_codes:
                break
            word = re.sub(r'(?<!\S)' + re.escape(' '.join(bigram)) + r'(?!\S)', ''.join(bigram), word)
        return word.split()

    @classmethod 
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        return Counter(tokenized)