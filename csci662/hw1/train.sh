python train.py -m logisticregression -i datasets/4dim.train.txt -o models/lr.4dim.model
python train.py -m logisticregression -i datasets/news.train.txt -o models/lr.news.model
# python train.py -m logisticregression -i datasets/products.train.txt -o models/lr.products.model
python train.py -m logisticregression -i datasets/questions.train.txt -o models/lr.questions.model

# python train.py -m naivebayes -i datasets/4dim.train.txt -o models/nb.4dim.model
# python train.py -m naivebayes -i datasets/news.train.txt -o models/nb.news.model
# python train.py -m naivebayes -i datasets/products.train.txt -o models/nb.products.model
# python train.py -m naivebayes -i datasets/questions.train.txt -o models/nb.questions.model