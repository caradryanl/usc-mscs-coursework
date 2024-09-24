from Model import *
import numpy as np
from Features import Features, ValidationFeatures

class MultiLabelLogisticRegressionModel:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes)
        self.biases = np.zeros(num_classes)
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class LogisticRegression(Model):
    def train(self, input_file, ):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        try:
            if '4dim' in input_file:
                with open('bpe/4dim', "rb") as file:
                    feature_checkpoint = pickle.load(file)
            elif 'news' in input_file:
                with open('bpe/news', "rb") as file:
                    feature_checkpoint = pickle.load(file)
            elif 'products' in input_file:
                with open('bpe/products', "rb") as file:
                    feature_checkpoint = pickle.load(file)
            elif 'questions' in input_file:
                with open('bpe/questions', "rb") as file:
                    feature_checkpoint = pickle.load(file)
            else:
                feature_checkpoint = {
                    'vocab': None,
                    'bpe_codes': None
                }
        except:
            feature_checkpoint = {
                    'vocab': None,
                    'bpe_codes': None
                }
        features = Features(input_file, bpe_codes=feature_checkpoint['bpe_codes'], vocab=feature_checkpoint['vocab'])
        
        vocab = {word: idx for idx, word in enumerate(features.vocab)}
        label_to_idx = {label: idx for idx, label in enumerate(features.labelset)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                    
        num_classes = len(features.labelset)
        model = MultiLabelLogisticRegressionModel(len(vocab), num_classes)
        
        # num_epochs = 50
        # batch_size = 64
        # lr = 0.01
        num_epochs = 25
        batch_size = 8
        lr = 0.01
        eps = 1e-15 
        length = len(features.tokenized_text)

        for e in range(num_epochs):
            losses = []
            print(f"===Epoch {e+1}===")
            
            # stochastic gradient descent, cannot converge without this
            indices = np.random.permutation(length)
            
            for i in range(0, length, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_inputs = []
                batch_targets = []

                for idx in batch_indices:
                    text, target = features.tokenized_text[idx], features.labels[idx]
                    target = label_to_idx[target]

                    bow = np.zeros(len(vocab))
                    for word in text:
                        bpe_tokens = features.apply_bpe(word)
                        for token in bpe_tokens:
                            if token in vocab:
                                bow[vocab[token]] += 1
                    
                    batch_inputs.append(bow)
                    batch_targets.append(target)

                batch_inputs = np.array(batch_inputs)
                batch_targets = np.eye(num_classes)[batch_targets]

                outputs = model.forward(batch_inputs)
                probs = model.sigmoid(outputs)

                probs = np.clip(probs, eps, 1 - eps)
                loss = -np.mean(np.sum(batch_targets * np.log(probs) + (1 - batch_targets) * np.log(1 - probs), axis=1))
                losses.append(loss)

                grad_outs = probs - batch_targets
                grad_w = np.dot(batch_inputs.T, grad_outs) / batch_size
                grad_b = np.mean(grad_outs, axis=0)
                model.weights -= lr * grad_w
                model.biases -= lr * grad_b

                
            print(f"    loss: {np.mean(np.array(losses))}")
        
        model_checkpoint = {
            'model': model,
            'vocab': features.vocab,
            'idx_to_label': idx_to_label,
            'bpe_codes': features.bpe_codes
        }
        ## Save the model
        self.save_model(model_checkpoint)
        return model_checkpoint

    def classify(self, input_file, model_checkpoint):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here (and change return)
        model = model_checkpoint['model']
        vocab = model_checkpoint['vocab']
        print(f"ckpt 1: {len(vocab)}")
        idx_to_label = model_checkpoint['idx_to_label']
        bpe_codes = model_checkpoint['bpe_codes']

        features = ValidationFeatures(input_file, bpe_codes=bpe_codes, vocab=vocab)
        
        preds = []
        vocab = {word: idx for idx, word in enumerate(features.vocab)}
        for text in features.tokenized_text:
            print(f"ckpt 2: {len(vocab)}")
            bow = np.zeros(len(vocab))
            for word in text:
                bpe_tokens = features.apply_bpe(word)
                for token in bpe_tokens:
                    if token in vocab:
                        bow[vocab[token]] += 1
            
            print(f"ckpt 3: {bow.shape}")
            output = model.forward(bow)
            predicted_label = np.argmax(output)
            preds.append(idx_to_label[predicted_label])
        
        # accuracy = np.mean([p == t for p, t in zip(preds, gts)])
        
        return preds



