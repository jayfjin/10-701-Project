import data
from sklearn import linear_model
import numpy as np

if __name__ == "__main__" or not os.file.exists('bow_selected.txt'):
    nonzero, = np.where(data.BBC_x.loc[:,data.isbow].std() != 0)
    BBC_nonzero = data.BBC_x.loc[:,data.isbow].iloc[:,nonzero]
    alphas_grid, scores_path = linear_model.lasso_stability_path(BBC_nonzero.values, data.BBC_y)

    selected = scores_path[:,99] != 0
    bow_selected = np.zeros(data.isbow.sum(), bool)
    bow_selected[nonzero[selected]] = True
    np.savetxt('bow_selected.txt', bow_selected, fmt='%d')

    # Borrowed from sklearn plot_sparse_recovery.py
    import pylab
    pylab.ion()
    sel = pylab.plot(alphas_grid[1:] ** .333, scores_path[selected,1:].T, 'r')
    nsel= pylab.plot(alphas_grid[1:] ** .333, scores_path[~selected,1:].T, 'k')
    pylab.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
    pylab.ylabel('Stability score: proportion of lasso models where feature selected')
    pylab.axis('tight')
    pylab.legend((sel[0], nsel[0]), ('selected features', 'other features'))
    pylab.savefig('lasso_select2.pdf')
