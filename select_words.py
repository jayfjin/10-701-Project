import sys
import numpy as np
import gzip
words = np.loadtxt('adblocktv/bow_selected.txt', dtype=bool)
keepfeats = np.zeros(4126, bool)
keepfeats[1:123] = True
keepfeats[np.where(words)[0] + 122] = True
keepfeats[4124:] = True
newfeatmap = np.cumsum(keepfeats) - 1
output = gzip.open(sys.argv[2], 'w')
for line in gzip.open(sys.argv[1]):
    feats = line.strip().split()
    def filter_feat(feat):
        i, value = feat.split(':')
        i = int(i)
        return keepfeats[i]
    def update_feat(feat):
        i, value = feat.split(':')
        i = int(i)
        return str(newfeatmap[i]) + ':' + str(value)
    feats = feats[0:1] + map(update_feat, filter(filter_feat, feats[1:]))
    output.write(' '.join(feats) + '\n')
output.flush()
output.close()
