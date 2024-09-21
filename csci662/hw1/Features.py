""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
from collections import Counter
import re
from tqdm import tqdm

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():    
        symbols = word.split()  # "h e l l o" -> "h" "e" "l" "l" "o"
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq # freq is the word freq. For each appearance of the word, add one pair freq for the pair.
    return pairs    

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        # print(f"w_out: {w_out} word: {word}")
        # print(w_out, word)
        v_out[w_out] = v_in[word]
    return v_out

def tokenize(text):
    # TODO customize to your needs
    text = text.lower()
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class Features:
    def __init__(self, data_file, num_merges=5000, bpe_codes=None, vocab=None):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))

        print(len(texts), len(self.labels))
        # print(texts[0], self.labels[0])

        self.tokenized_text = [tokenize(text) for text in texts]
        self.labelset = list(set(self.labels))

        # print(self.tokenized_text[0], self.labelset)
        
        # BPE
        if bpe_codes != None and vocab != None:
            self.bpe_codes = bpe_codes
            self.vocab = vocab
        else:
            vocab = Counter(word for text in self.tokenized_text for word in text)  # {"hello": 3}
            vocab = {' '.join(word): count for word, count in vocab.items()}    # {"h e l l o": 3}
            self.bpe_codes = {}
            for i in tqdm(range(num_merges), desc="BPE Merging..."):
                pairs = get_stats(vocab)
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                # print(best)
                vocab = merge_vocab(best, vocab)
                self.bpe_codes[best] = i

            
            self.vocab = set(vocab.keys())
        # print(self.vocab)

    def __len__(self):
        return len(self.tokenized_text)

    def apply_bpe(self, word):
        '''
            Return the correct word split according to the self.bpe_codes
        '''
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
    def get_features(tokenized):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        '''
            Note: We put featuring processes in the model so no get_features implementation here.
        '''
        return Counter(tokenized)
    
class ValidationFeatures:
    def __init__(self, data_file, num_merges=5000, bpe_codes=None, vocab=None):
        with open(data_file) as file:
            data = file.read().splitlines()
        print(data[0])

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts = map(list, zip(*data_split))

        print(texts[0])

        self.tokenized_text = [tokenize(text) for text in texts]
        
        # BPE
        if bpe_codes != None and vocab != None:
            self.bpe_codes = bpe_codes
            self.vocab = vocab
        else:
            vocab = Counter(word for text in self.tokenized_text for word in text)  # {"hello": 3}
            vocab = {' '.join(word): count for word, count in vocab.items()}    # {"h e l l o": 3}
            self.bpe_codes = {}
            for i in tqdm(range(num_merges), desc="BPE Merging..."):
                pairs = get_stats(vocab)
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                # print(best)
                vocab = merge_vocab(best, vocab)
                self.bpe_codes[best] = i

            
            self.vocab = set(vocab.keys())
        # print(self.vocab)

    def __len__(self):
        return len(self.tokenized_text)

    def apply_bpe(self, word):
        '''
            Return the correct word split according to the self.bpe_codes
        '''
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