"""
 Refer to Chapter 5 for more details on how to implement a LogisticRegression
"""
from Model import *
from Features import Features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, bpe_codes):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.bpe_codes = bpe_codes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        bow = torch.zeros(len(self.vocab))
        for word in text:
            bpe_tokens = self.apply_bpe(word)
            for token in bpe_tokens:
                if token in self.vocab:
                    bow[self.vocab[token]] += 1
        
        return bow, torch.tensor(label)

    def apply_bpe(self, word):
        return Features.apply_bpe(self, word)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class LogisticRegression(Model):
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        features = Features(input_file)
        
        vocab = {word: idx for idx, word in enumerate(features.vocab)}
        label_to_idx = {label: idx for idx, label in enumerate(features.labelset)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        labels = [label_to_idx[label] for label in features.labels]
        
        dataset = TextDataset(features.tokenized_text, labels, vocab, features.bpe_codes)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = LogisticRegressionModel(len(vocab), len(features.labelset))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        num_epochs = 10
        for _ in range(num_epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        model_info = {
            'model': model,
            'vocab': vocab,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'bpe_codes': features.bpe_codes
        }

        ## Save the model
        self.save_model(model_info)
        return model_info

    def classify(self, input_file, model_info):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here (and change return)
        features = Features(input_file)
        
        model = model_info['model']
        vocab = model_info['vocab']
        idx_to_label = model_info['idx_to_label']
        bpe_codes = model_info['bpe_codes']
        
        preds = []
        model.eval()
        with torch.no_grad():
            for text in features.tokenized_text:
                bow = torch.zeros(len(vocab))
                for word in text:
                    bpe_tokens = features.apply_bpe(word)
                    for token in bpe_tokens:
                        if token in vocab:
                            bow[vocab[token]] += 1
                
                output = model(bow)
                _, predicted = torch.max(output, 0)
                preds.append(idx_to_label[predicted.item()])
        
        return preds