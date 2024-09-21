python classify.py -m models/lr.4dim.model -i datasets/4dim.train.txt -o outputs/eval.lr.4dim.txt
python classify.py -m models/lr.news.model -i datasets/news.train.txt -o outputs/eval.lr.news.txt
python classify.py -m models/lr.products.model -i datasets/products.train.txt -o outputs/eval.lr.products.txt
python classify.py -m models/lr.questions.model -i datasets/questions.train.txt -o outputs/eval.lr.questions.txt

python classify.py -m models/nb.4dim.model -i datasets/4dim.train.txt -o outputs/eval.nb.4dim.txt
python classify.py -m models/nb.news.model -i datasets/news.train.txt -o outputs/eval.nb.news.txt
python classify.py -m models/nb.products.model -i datasets/products.train.txt -o outputs/eval.nb.products.txt
python classify.py -m models/nb.questions.model -i datasets/questions.train.txt -o outputs/eval.nb.questions.txt