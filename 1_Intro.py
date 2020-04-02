#!/usr/bin/python3.2
#-*- coding: utf-8 -*-
import os
import rpy2.robjects as robjects
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class _Intro():
    def __init__(self):
        robjects.r['load']("../inst/extdata/cyto.RData")
        rcyto = robjects.r['cyto']
        #print(cyto.r_repr())
        #print(cyto)        

        cyto = np.array(rcyto)

        # number of rows and columns in the cyto objec
        nrows = np.size(cyto, 0)
        print(nrows)
        ncols = np.size(cyto, 1)
        print(ncols)

        # A specific column of a matrix can be selected by its number:
        fcs_a = cyto[:, 0]
        #print(fcs_a)

        #we can select a specific row:
        cell2 =  cyto[1, ]
        #print(cell2)

        # cytoEmb = TSNE(n_components=2).fit_transform(cyto)
        cytoEmb = TSNE(n_components=2, perplexity=2).fit_transform(cyto)

        print(cytoEmb)
        df = pd.DataFrame(cytoEmb, columns = ['tsne1', 'tsne2'])

        # for marker in ['o', 'x']:
        #    plt.plot(cytoEmb[:, 0], cytoEmb[:, 1], marker,
        #            label="marker='{0}'".format(marker))
        #plt.legend(numpoints=1)
        #plt.xlim(0, 1.8);

        df.to_excel(r'Resultats/1_Intro/tsne_result.xlsx', index = False)

        
        dfs = pd.read_excel('Resultats/1_Intro/tsne_result.xlsx', index_col=None)
        plt.xlabel("tsne1")
        plt.ylabel("tsne2")
        plt.title("tsne result sur les Cyto data perplexity=2")
        plt.scatter(dfs.iloc[:, 0], dfs.iloc[:, 1])
        plt.savefig("Resultats/1_Intro/tsne_2perplexity.png")
        plt.show()




def main():
    _Intro()

if __name__ == '__main__':
    main()