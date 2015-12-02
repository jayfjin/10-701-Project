from sklearn.datasets import load_svmlight_file
import gzip
import re
import pandas as pd

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
    elif not line:
        isfeats = False
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
                for i in xrange(int(inds.group(1)), int(inds.group(2))+1):
                    featnames.append(featname + str(i))
            else:
                featnames.append(featname)
            featname = None

BBC_x = pd.DataFrame(BBC_x.toarray(), columns=featnames)
CNN_x = pd.DataFrame(CNN_x.toarray(), columns=featnames)
CNNIBN_x = pd.DataFrame(CNNIBN_x.toarray(), columns=featnames)
NDTV_x = pd.DataFrame(NDTV_x.toarray(), columns=featnames)
TIMESNOW_x = pd.DataFrame(TIMESNOW_x.toarray(), columns=featnames)

isbow = pd.Series(featnames, index=featnames).str.contains('BagofAudioWords')
