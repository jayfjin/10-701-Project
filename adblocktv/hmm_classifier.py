from hmmlearn import hmm
import numpy as np

def build_hmms(X, Y, n_states=32):
    split = np.zeros(len(X))
    split[1:] = Y[1:] != Y[:-1]
    # Points to split X into positive/negative example blocks
    split = list(np.where(Y[1:] != Y[:-1])[0] + 1)
    plus_blocks = []
    minus_blocks = []
    for start, end in zip([0] + split, split + [None]):
        block = X.values[start:end]
        if Y[start] > 0:
            plus_blocks.append(block)
        else:
            minus_blocks.append(block)
    plus_hmm = hmm.GaussianHMM(n_states)
    minus_hmm = hmm.GaussianHMM(n_states)
    plus_hmm.fit(np.concatenate(plus_blocks))
    minus_hmm.fit(np.concatenate(minus_blocks))
    # Transition matrices are sometimes screwed up
    plus_hmm.transmat_ /= plus_hmm.transmat_.sum(1)[:, None]
    minus_hmm.transmat_ /= minus_hmm.transmat_.sum(1)[:, None]
    return plus_hmm, minus_hmm

def predict(hmms, X, window_size=16):
    X = X.values
    plus_hmm, minus_hmm = hmms
    pred = np.zeros(len(X), int)
    for i in xrange(len(X)):
        window_start = max(0, i - window_size)
        window = X[window_start:i+1]
        if plus_hmm.score(window) > minus_hmm.score(window):
            pred[i] = 1
        else:
            pred[i] = -1
    return pred
