"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""

from Model import *
from Features import Features
from collections import defaultdict
import math

class NaiveBayes(Model):
    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        features = Features(input_file)
        model = {
            'class_counts': defaultdict(int),
            'word_counts': defaultdict(lambda: defaultdict(int)),
            'vocab': features.vocab,
            'labelset': features.labelset,
            'bpe_codes': features.bpe_codes,
            'total_words': defaultdict(int)
        }

        for tokens, label in zip(features.tokenized_text, features.labels):
            model['class_counts'][label] += 1
            for word in tokens:
                bpe_tokens = features.apply_bpe(word)
                for token in bpe_tokens:
                    model['word_counts'][label][token] += 1
                    model['total_words'][label] += 1

        ## Save the model
        self.save_model(model)
        return model

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 
        ## TODO write your code here
        features = Features(input_file)
        
        preds = []
        for text in features.tokenized_text:
            scores = defaultdict(float)
            for label in model['labelset']:
                scores[label] = math.log(model['class_counts'][label])
                for word in text:
                    bpe_tokens = features.apply_bpe(word)
                    for token in bpe_tokens:
                        word_count = model['word_counts'][label].get(token, 0) + 1  # Laplace smoothing
                        total_words = model['total_words'][label] + len(model['vocab'])
                        scores[label] += math.log(word_count / total_words)
            
            best_label = max(scores, key=scores.get)
            preds.append(best_label)

        return preds


