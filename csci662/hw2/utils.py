# put in relevant import statements here
import random
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def example_transform(examples):
    res= None

    res = {"text": [x.upper() for x in examples["text"]]}

    return res

### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
def typos(sentence):
    words = sentence.split()
    typo_sentence = []
    for word in words:
        if random.random() < 0.25:
            typo_sentence.append(generate_typos(word))
        else:
            typo_sentence.append(word)
    return " ".join(typo_sentence)

def generate_typos(word):
    # basically exchange the place of 
    if len(word) > 2:
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    return word

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)

        # directly make the replacement probability 1.0 for extreme performance degradation
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_sentence.append(synonym if synonym != word else word)
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)


def custom_transform(examples):

    res = {"text": []}
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update each item in examples["text"] using your transformation, as is done in the example above    
    for sentence in examples["text"]:
        # Apply synonym replacement
        sentence = synonym_replacement(sentence)
        
        # Apply typo introduction
        sentence = typos(sentence)
        
        res["text"].append(sentence)
    ##### YOUR CODE ENDS HERE ######

    return res
