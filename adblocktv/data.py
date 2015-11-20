from sklearn.datasets import load_svmlight_file

BBC_x, BBC_y = load_svmlight_file(gzip.open("../data/BBC.txt.gz"))
CNN_x, CNN_y = load_svmlight_file(gzip.open("../data/CNN.txt.gz"))
CNNIBN_x, CNNIBN_y = load_svmlight_file(gzip.open("../data/CNNIBN.txt.gz"))
NDTV_x, NDTV_y = load_svmlight_file(gzip.open("../data/NDTV.txt.gz"))
TIMESNOW_x, TIMESNOW_y = load_svmlight_file(gzip.open("../data/TIMESNOW.txt.gz"))
