from abc import ABCMeta, abstractmethod
import pickle


PROMPT = """
You are a question answerer based on the given retrieval documents. The answer will not be too long and you should find the answer in the documents then focus simple key words. Just keep the first three answers.
Documents: {retrieved_documents}. Question: {questions}. Answers:
"""

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def load_model(self):
        # this method will be necessary if you're using Huggingface `generate`,
        # but it is not necessary for Ollama.
        pass 

    @abstractmethod
    def query(self, retrieved_documents, questions):
        pass