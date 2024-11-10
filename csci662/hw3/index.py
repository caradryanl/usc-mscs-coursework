import pickle
import argparse
from bm25 import *


def get_arguments():
    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", help="retriever model: what retriever to use")
    parser.add_argument("-i",  help="inputfile: the name/path of the file to index; it has to be read one text per line")
    parser.add_argument("-n", help="index_name: the name/path of the index (you should write it on disk)")
   

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if "bm25" in args.m:
        model = BM25(model_file=args.m)
    else:
        ## TODO Add any other models you wish to evaluate
        model = None
        
    index = model.index(args.i)
    
    ## Save the index
    with open(args.o, "wb") as file:
        pickle.dump(index, file)