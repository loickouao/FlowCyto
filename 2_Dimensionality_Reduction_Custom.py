#!/usr/bin/python3.2
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import flowio
import flowutils
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

class _Dimensionality_Reduction():
    def __init__(self):
        fd = flowio.FlowData('../inst/extdata/21-10-15_Tube_028_Live.fcs')
        #print(fd.text['spillover'])
        data = np.reshape(fd.events, (-1, fd.channel_count))
        spill, markers = flowutils.compensate.get_spill(fd.text['spillover'])
        print(data.shape)
        #print(spill.shape)
        #print(markers)
        #print(events)
        #print(fd.channels)


        indices_markers = []
        idx_channel = []
        select_channel = ['PE-A', 'PE-Cy5-A', 'PE-Cy7-A']
        for channel in fd.channels:
            if fd.channels[channel]['PnN'] in markers:
                indices_markers.append(int(channel) - 1)
                if fd.channels[channel]['PnN'] in select_channel:
                    idx_channel.append(int(channel) - 1)

        data_select = data[:, indices_markers]
        comp_result = np.linalg.solve(spill.T, data_select.T).T
        # Vérification de la solution
        verif = np.allclose(np.dot(spill.T,  comp_result.T), data_select.T)
        print(verif)

        data_comp = data.copy()
        data_comp[:, indices_markers] = comp_result
        #print(data_comp[:, idx_channel])

        


        '''These values also need to be transformed.
        The logicle transformation takes several parameters, we provide the indices of 
        the columns that we want to transform:
        '''

        xform_data = flowutils.transforms.logicle(data_comp, indices_markers)
        xform_data_select = xform_data[0:3, idx_channel]
        xform_df = pd.DataFrame(xform_data_select, columns = select_channel)
        print(xform_df)

        'log transformation'
        logform_data = flowutils.transforms.hyperlog(data_comp, indices_markers)
        logform_data_select = logform_data[0:3, idx_channel]
        logform_df = pd.DataFrame(logform_data_select, columns = select_channel)
        print(logform_df)



        # Create four polar axes and access them through the returned array
        data_plot = data[0:5000, idx_channel[0:3]]
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 10.5)
        axs[0, 0].scatter(data_plot[:, 0], data_plot[:, 1])
        axs[0, 0].set_title("no compensate and no transform data on 5000 cells")
        #axs[0, 0].set_xlabel('PE-A')
        axs[0, 0].set_ylabel('PE-Cy5-A')
        data_plot = data_comp[0:5000, idx_channel[0:3]]
        axs[0, 1].scatter(data_plot[:, 0], data_plot[:, 1])
        axs[0, 1].set_title(" compensate and no transform data on 5000 cells")
        #axs[0, 1].set_xlabel('PE-A')
        axs[0, 1].set_ylabel('PE-Cy5-A')
        data_plot = xform_data[0:5000, idx_channel[0:3]]
        axs[1, 0].scatter(data_plot[:, 0], data_plot[:, 1])
        axs[1, 0].set_title(" compensate and logicle transformation on 5000 cells")
        axs[1, 0].set_xlabel('PE-A')
        axs[1, 0].set_ylabel('PE-Cy5-A')
        data_plot = logform_data[0:5000, idx_channel[0:3]]
        axs[1, 1].scatter(data_plot[:, 0], data_plot[:, 1])
        axs[1, 1].set_title(" compensate and log transformation on 5000 cells")
        axs[1, 1].set_xlabel('PE-A')
        axs[1, 1].set_ylabel('PE-Cy5-A')
        plt.savefig("Resultats/2_Dimensionality_Reduction/compensate_transform.png")
        plt.show()
        


        # tSNE -------------------------------------
        
        idx_interest = [7] + list(range(9,14)) + list(range(15,20))

        subsample = xform_data[0:1000, ] 
        
        subsampleEmb = TSNE(n_components=2).fit_transform(subsample)
        
        subsample2 = subsample[:, idx_interest]
        subsampleEmb2 = TSNE(n_components=2).fit_transform(subsample2)
        
        #Run tSNE on the same subsample of cells, but this time using only the "CD3" "CD11c" and "CD19" channels.
        subsample3 = subsample[:, idx_channel]
        subsampleEmb3 = TSNE(n_components=2).fit_transform(subsample3)

        #Run tSNE on the same subsample of cells, but this time using only the "CD3" and "CD19" channels.
        subsample4 = subsample[:, idx_channel[0:3]]
        subsampleEmb4 = TSNE(n_components=2).fit_transform(subsample4)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 10.5)
        axs[0, 0].scatter(subsampleEmb[:, 0], subsampleEmb[:, 1])
        axs[0, 0].set_title("tsne data of all channels(19) on 1000 cells")
        #axs[0, 0].set_xlabel('tsne1')
        axs[0, 0].set_ylabel('tsne2')

        axs[0, 1].scatter(subsampleEmb2[:, 0], subsampleEmb2[:, 1])
        axs[0, 1].set_title("tsne data of 11 channels on 1000 cells")
        #axs[0, 1].set_xlabel('tsne1')
        axs[0, 1].set_ylabel('tsne2')

        axs[1, 0].scatter(subsampleEmb3[:, 0], subsampleEmb3[:, 1])
        axs[1, 0].set_title("tsne data of 3 channels on 1000 cells")
        axs[1, 0].set_xlabel('tsne1')
        axs[1, 0].set_ylabel('tsne2')

        axs[1, 1].scatter(subsampleEmb4[:, 0], subsampleEmb4[:, 1])
        axs[1, 1].set_title("tsne data of PE-A and PE-Cy5-A channels on 1000 cells")
        axs[1, 1].set_xlabel('tsne1')
        axs[1, 1].set_ylabel('tsne2')
        plt.savefig("Resultats/2_Dimensionality_Reduction/tsne.png")
        plt.show()

        
        #PCA ------------------------------------

        #channels_of_interest PE-A  PE-Cy5-A
        subsample_of_interest = subsample[:, idx_channel[0:3]]

        pca = PCA(n_components=2)
        PCS = pca.fit_transform(subsample_of_interest)      
        df = pd.DataFrame(PCS, columns = ['PC1', 'PC2'])  
        var_PCS = pca.explained_variance_ratio_
        print(var_PCS)
        namebars = ['PC1', 'PC2']
        y_pos = np.arange(len(namebars))

        '''
        # Create bars
        plt.bar(y_pos, var_PCS)
        plt.xticks(y_pos, namebars)
        plt.ylabel("Variances")
        plt.title("pca")
        # Show graphic
        plt.savefig("Resultats/2_Dimensionality_Reduction/pca_variance.png")
        plt.show()

        ax1 = df.plot.scatter(x='PC1', y='PC2', c='DarkBlue')
        ax1.set_title("pca for channels PE-A  PE-Cy5-A on 1000 cells")
        plt.savefig("Resultats/2_Dimensionality_Reduction/pca_scatter.png")
        plt.show()
        '''
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5)
        ax[0].bar(y_pos, var_PCS)
        #ax[0].set_xticks(y_pos, namebars)
        ax[0].set_xticks(y_pos)
        ax[0].set_xticklabels(namebars)
        #ax[0].set_xticklabels(map(str, namebars))
        ax[0].set_ylabel("Variances")
        ax[0].set_title("pca")

        ax[1].scatter(PCS[:, 0], PCS[:, 1], c='DarkBlue')
        ax[1].set_xlabel("PC1")
        ax[1].set_ylabel("PC2")
        ax[1].set_title("pca for channels PE-A  PE-Cy5-A on 1000 cells")
        plt.savefig("Resultats/2_Dimensionality_Reduction/pca.png")
        plt.show()

        # MDS ---------------------------------

        # par defaut distance euclidienne
        mds = MDS(n_components=2)
        
        embedding = mds.fit_transform(subsample_of_interest)
        x, y = embedding[:, 0], embedding[:, 1]

        fig, ax = plt.subplots()
        ax.set_title("MDS for channels PE-A  PE-Cy5-A on 1000 cells")
        ax.set_xlabel('EMB1')
        ax.set_ylabel('EMB2')
        ax.scatter(x, y)
        plt.savefig("Resultats/2_Dimensionality_Reduction/mds.png")
        plt.show()
        
        
        
def main():
    _Dimensionality_Reduction()

if __name__ == '__main__':
    main()