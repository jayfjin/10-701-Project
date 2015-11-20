from sklearn.datasets import load_svmlight_file
import gzip
import re

BBC_x, BBC_y = load_svmlight_file(gzip.open("../data/BBC.txt.gz"))
CNN_x, CNN_y = load_svmlight_file(gzip.open("../data/CNN.txt.gz"))
CNNIBN_x, CNNIBN_y = load_svmlight_file(gzip.open("../data/CNNIBN.txt.gz"))
NDTV_x, NDTV_y = load_svmlight_file(gzip.open("../data/NDTV.txt.gz"))
TIMESNOW_x, TIMESNOW_y = load_svmlight_file(gzip.open("../data/TIMESNOW.txt.gz"))

featnames = []
readme = open('../data/readme.txt')
isfeats = False
featname = None
for line in readme:
    line = line.strip()
    if line == "Dimension Index in feature File":
        isfeats = True
    elif isfeats:
        if featname is None:
            words = line.split()
            lastword = None
            try:
                lastword = words.index('(')
            except ValueError:
                pass
            featname = ''.join(words[:lastword])
        else:
            inds = re.match(r'([0-9]+)\s+-\s+([0-9]+)', line)
            if inds:
                for i in xrange(int(inds.group(1)), int(inds.group(2))):
                    featnames.append(featname + str(i))
            else:
                featnames.append(featname)
            featname = None
