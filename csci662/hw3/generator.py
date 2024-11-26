import argparse
from ollama import *
from bm25 import * 
from tfidf import *
from hgf import HuggingfaceModel
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description="Generator")
    parser.add_argument('-r', default="bm25", help="name of the retriever model you used.")
    parser.add_argument("-n", help="the name of the index that you created with 'index.py'.")
    parser.add_argument("-k", help="the number of documents to return in each retrieval run.")

    parser.add_argument("-p", default="ollama", help="Platform to use for the generator.")
    parser.add_argument("-m", help="type of model to use to generate: gemma2:2b, etc.")
    
    parser.add_argument("-i", help="path of the input file of questions, where each question is in the form: <text> for each newline")
    parser.add_argument("-o", help="path of the file where the answers should be written") # Respect the naming convention for the model: make sure to name it *.answers.txt in your workplace otherwise the grading script will fail

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.r == "bm25":
        retriever = BM25(args.n).load_model()
    elif args.r ==  "tfidf":
        retriever = TFIDF(args.n).load_model()
    else:
        retriever = None

    if args.p == 'ollama':
        generator = OllamaModel(model_name=args.m)
    elif args.p == 'huggingface':
        generator = HuggingfaceModel(model_name=args.m)
    else:
        ## TODO Add any other models you wish to train
        generator = None

    answers = []

    with open(args.i) as f:
        # Skip first line if it's just a number
        first_line = f.readline().strip()
        if first_line.isdigit():
            starting_pos = f.tell()
        else:
            f.seek(0)
            starting_pos = 0
        f.seek(starting_pos)
        questions = f.read().strip().splitlines()
    count = 0 
    # Using tqdm with a descriptive progress bar
    for q in tqdm(questions, desc="Processing Questions", total=len(questions)):
        q = q.strip()
        docs = retriever.search(q, args.k)
        answer = generator.query(docs, q)
        answers.append(answer)
        count += 1
        if count >= 3:
            break
        
    with open(args.o, 'w') as f:
        for a in answers:
            f.write(a)
            f.write('\n')






