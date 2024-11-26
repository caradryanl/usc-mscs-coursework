python generator.py -r bm25 -n bm25 -k 0 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/bm25_0_gemma_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 0 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/bm25_0_gemma_test.answers.txt
python generator.py -r bm25 -n bm25 -k 2 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/bm25_2_gemma_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 2 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/bm25_2_gemma_test.answers.txt
python generator.py -r bm25 -n bm25 -k 4 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/bm25_4_gemma_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 4 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/bm25_4_gemma_test.answers.txt

python generator.py -r tfidf -n tfidf -k 0 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/tfidf_0_gemma_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 0 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/tfidf_0_gemma_test.answers.txt
python generator.py -r tfidf -n tfidf -k 2 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/tfidf_2_gemma_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 2 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/tfidf_2_gemma_test.answers.txt
python generator.py -r tfidf -n tfidf -k 4 -p huggingface -m google/gemma-2b -i datasets/question.dev.txt -o outputs/tfidf_4_gemma_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 4 -p huggingface -m google/gemma-2b -i datasets/question.test.txt -o outputs/tfidf_4_gemma_test.answers.txt

python generator.py -r bm25 -n bm25 -k 0 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/bm25_0_llama_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 0 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/bm25_0_llama_test.answers.txt
python generator.py -r bm25 -n bm25 -k 2 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/bm25_2_llama_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 2 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/bm25_2_llama_test.answers.txt
python generator.py -r bm25 -n bm25 -k 4 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/bm25_4_llama_dev.answers.txt
python generator.py -r bm25 -n bm25 -k 4 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/bm25_4_llama_test.answers.txt

python generator.py -r tfidf -n tfidf -k 0 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/tfidf_0_llama_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 0 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/tfidf_0_llama_test.answers.txt
python generator.py -r tfidf -n tfidf -k 2 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/tfidf_2_llama_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 2 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/tfidf_2_llama_test.answers.txt
python generator.py -r tfidf -n tfidf -k 4 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.dev.txt -o outputs/tfidf_4_llama_dev.answers.txt
python generator.py -r tfidf -n tfidf -k 4 -p huggingface -m unsloth/Llama-3.2-1B-Instruct -i datasets/question.test.txt -o outputs/tfidf_4_llama_test.answers.txt