import argparse
from naivebayes import *
from logisticregression import *


def get_arguments():
    parser = argparse.ArgumentParser(description="Text Classifier Trainer")
    parser.add_argument("-i", help="path of the input file where training file is in the form <text>TAB<label>")
    parser.add_argument("-o", help="path of the file where the model is saved") # Respect the naming convention for the model: make sure to name it {nb, perceptron}.{4dim, authors, odiya, products}.model for your best models in your workplace otherwise the grading script will fail

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    features = Features(args.i)
    
    feature_checkpoint = {
        'vocab': features.vocab,
        'bpe_codes': features.bpe_codes
    }
    ## Save the model
    with open(args.o, "wb") as file:
        pickle.dump(feature_checkpoint, file)