from GeneratorModel import *
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingfaceModel(GeneratorModel):
    def __init__(self, model_name, max_words=6000):
        self.model_name = model_name
        self.max_words = max_words

        self.load_model(model_name)

    def load_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def clean_text(self, text):
        """
        Clean and format the generated text into a single line
        """
        # Replace any newlines, multiple spaces, and trim
        cleaned = ' '.join(text.split())
        # Remove any remaining special characters that might cause line breaks
        cleaned = cleaned.replace('\r', ' ').replace('\t', ' ')
        return cleaned.strip()

    def truncate_documents(self, documents):
        """
        Truncate the combined documents to stay within max_words limit.
        Returns the truncated combined string.
        """
        total_words = 0
        truncated_docs = []
        
        for doc in documents:
            words = doc.split()
            remaining_words = self.max_words - total_words
            
            if total_words >= self.max_words:
                break
                
            if len(words) <= remaining_words:
                truncated_docs.append(doc)
                total_words += len(words)
            else:
                # Take only the remaining number of words needed
                truncated_doc = ' '.join(words[:remaining_words])
                truncated_docs.append(truncated_doc)
                total_words = self.max_words
                break
                
        return '\n'.join(truncated_docs)

    def query(self, retrieved_documents, questions):
        # Truncate documents if they exceed max_words
        truncated_documents = self.truncate_documents(retrieved_documents)
        
        # Create prompt with truncated documents
        prompt = PROMPT.format(retrieved_documents=truncated_documents, questions=questions)

        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]

        # Generate with specific parameters for better control
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the new tokens by slicing the output
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return self.clean_text(generated_text)