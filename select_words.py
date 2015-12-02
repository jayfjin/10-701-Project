import sys
import numpy
import gzip
words = numpy.loadtxt('adblocktv/bow_selected.txt', dtype=bool)
wordfeats = list(122 + numpy.where(words)[0])
output = gzip.open(sys.argv[2], 'w')
for line in gzip.open(sys.argv[1]):
    feats = line.strip().split()
    def filter_feat(feat):
        i, value = feat.split(':')
        i = int(i)
        return i < 122 or i in wordfeats or i >= 4123
    def update_feat(feat):
        i, value = feat.split(':')
        i = int(i)
        if i < 122:
            return feat
        elif i < 4123:
            return str(122 + wordfeats.index(i)) + ':' + str(value)
        else:
            return str(i + len(wordfeats) - 4001) + ':' + str(value)
    feats = feats[0:1] + map(update_feat, filter(filter_feat, feats[1:]))
    output.write(' '.join(feats) + '\n')
output.flush()
output.close()
