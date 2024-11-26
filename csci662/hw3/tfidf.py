"""
 Refer to: https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus
 for more information
"""
import math
import re
from RetrievalModel import *
from typing import List, Dict


class TFIDF(RetrievalModel):
    def __init__(self, model_file):
        super().__init__(model_file)
        self.doc_freqs = {}  
        self.term_freqs = {} 
        self.doc_lens = {}  
        self.avg_doc_len = 0  
        self.N = 0  
        self.idf = {}  
        self.documents = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and split on whitespace
        text = text.lower()
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def index(self, input_file: str):
        """Index the tab-separated document file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            # Skip first line if it's just a number
            first_line = f.readline().strip()
            if first_line.isdigit():
                starting_pos = f.tell()
            else:
                f.seek(0)
                starting_pos = 0
            
            # Process each document
            f.seek(starting_pos)
            for doc_id, passage in enumerate(f):
                # Split on first tab to get doc_id and content
                parts = passage.strip().split('\t', 1)
                if len(parts) < 2:  # Skip malformed lines
                    continue
                    
                text = parts[1]  # Get the actual content
                self.documents[doc_id] = text
                
                # Tokenize the content
                tokens = self.tokenize(text)
                self.doc_lens[doc_id] = len(tokens)
                
                # Count term frequencies for this document
                term_freqs = {}
                for token in tokens:
                    if token not in term_freqs.keys():
                        term_freqs[token] = 1
                    else:
                        term_freqs[token] = term_freqs[token] + 1
                
                # Store term frequencies for this doc
                self.term_freqs[doc_id] = term_freqs
                
                # Update document frequencies
                for term in set(tokens):
                    if token not in self.doc_freqs.keys():
                        self.doc_freqs[token] = 1
                    else:
                        self.doc_freqs[token] = self.doc_freqs[token] + 1

        # Calculate aggregates
        self.N = len(self.doc_lens)
        if self.N > 0:
            self.avg_doc_len = sum(self.doc_lens.values()) / self.N

            # Calculate IDF scores
            for term, df in self.doc_freqs.items():
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                self.idf[term] = idf
                print(term, idf, df, self.N)

        self.save_model()

    def tfidf(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        
        for term in query_terms:
            term_freqs = {} if doc_id not in self.term_freqs.keys() else self.term_freqs[doc_id]
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = 0 if term not in self.idf.keys() else self.idf[term]
            
            score += idf * tf
            
        return score

    def search(self, query: str, k: int = 10) -> List[str]:
        """Search indexed documents and return top k document contents."""
        query_terms = self.tokenize(query)
        scores = []
        
        # Score all documents
        for doc_id in self.doc_lens.keys():
            score = self.tfidf(query_terms, doc_id)
            scores.append((doc_id, score))
            
        # Sort by score descending and return top k documents
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [self.documents[doc_id] for doc_id, _ in scores[0:int(k)]]
        return top_docs