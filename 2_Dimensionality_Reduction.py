#!/usr/bin/python3.2
#-*- coding: utf-8 -*-
from LoadFcs import *
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

class _Dimensionality_Reduction():
    def __init__(self, namefile):
        ff = Loadfcs(namefile)
        print("#In this matrix, each row represents a cell and every column represents a marker.\n #Number of cells") 
        print(ff.fd.event_count)

        print("\n#Notice that the column names correspond to the detector names in your machine.\n")
        indices = ff.get_all_indices()
        channels = ff.get_channels_by_indice(indices)
        print(channels)

        print("\n#We can make use of some functions from the FlowSOM package to easily access the actual marker names.\n")
        print(ff.get_markers( "BV605-A"))

        print("\n#This also works on a vector instead of a single value :\n")
        print(ff.get_markers(channels))

        print("\n#We also have a function available to do the reverse :\n")
        print(ff.get_channels("CD3"))

        print("\n#To access the actual matrix, we need to look at the exprs element of the flowframe \n")
        print("#object. Note that these values still need to be compensated and transformed! \n")
        df_head = pd.DataFrame(ff.data[0:6,], columns = channels)  
        print(df_head)

        print("\n \n#Additionally, all metadata describing the fcs file is also stored in the flowframe, in the description element.\n")
        #print(ff.fd.text)

        print("\n\n #One of the elements in the metadata description is the compensation matrix\n")
        comp = ff.fd.text['spillover']
        spill, spill_channels = flowutils.compensate.get_spill(comp) 
        df_comp = pd.DataFrame(spill, columns = spill_channels)  
        #print(df_comp)

        print("\n\n #We can use this compensation matrix to compensate the fcs file \n")
        channels_select = ff.get_channels(["CD3", "CD19", "CD11c"])
        indices_select = ff.get_indice(channels_select)
        df_select = pd.DataFrame(ff.data[0:3, indices_select], columns = channels_select)  
        print(df_select)
        ff_comp = ff.compensate()
        df_comp = pd.DataFrame(ff_comp[0:3, indices_select], columns = channels_select)  
        print(df_comp)

        
        print("\n# These values also need to be transformed.\n")
        print("The logicle transformation takes several parameters, we provide the indices of the columns that we want to transform: \n") 

        indices_channels = np.array(indices)-1
        
        tlogicle_data = ff.transform(ff_comp, indices_channels, type_transform="logicle")
        tlogicle_df_select = pd.DataFrame(tlogicle_data[0:3, indices_select], columns = channels_select)  
        print(tlogicle_df_select)
        
        data_plot = tlogicle_data[0:5000, indices_select[0:3]]
        plt.xlabel("PE-A")
        plt.ylabel("PE-Cy5-A")
        plt.title("compensate and logicle transformation on 5000 cells")
        plt.scatter(data_plot[:, 0], data_plot[:, 1])
        plt.savefig("Resultats/2_Dimensionality_Reduction/Figure_1 - first 5000 cells.png")
        plt.show()


        print("\n\n# tSNE")
        print("As tSNE can take a long time to run on large amounts of cells, we will work only on the 1000 first cells from the flowframe: \n")
        subsample = tlogicle_data[1:1000, ]
        print("We will also select the channels which we are interested in to perform the tSNE analysis:\n")
        idx_interest = [7] + list(range(9,14)) + list(range(15,20))
        channels_of_interest = ff.get_channels_by_indice(idx_interest)
        markers_of_interest = ff.get_markers(channels_of_interest)
        print(channels_of_interest)
        print(markers_of_interest)

        subsampleEmb = TSNE(n_components=2).fit_transform(subsample[:, idx_interest])
        plt.scatter(subsampleEmb[:, 0], subsampleEmb[:, 1])
        plt.title("tsne 1000 cells data of channels of_interest")
        plt.xlabel('tsne1')
        plt.ylabel('tsne2')
        plt.savefig("Resultats/2_Dimensionality_Reduction/Figure_2 - tsne 1000 cells of channels of_interest.png")
        plt.show()
        
        print("\nWe can also color this plot according to the expression values of a certain marker of interest:\n")
        
        df_subsample = pd.DataFrame(subsample, columns = ff.get_markers(channels))  
        df_subsampleEmb = pd.DataFrame(subsampleEmb, columns = ['tsne1', 'tsne2'])

        df_result = pd.concat([df_subsampleEmb, df_subsample], axis=1, sort=False)
        df_result.to_csv('Resultats/2_Dimensionality_Reduction/2_result1_Dimension_Reduction.csv', index=False)

        #Run tSNE on the same subsample of cells, but this time using only the "CD3" "CD11c" and "CD19" channels.
        channels_of_interest = ff.get_channels(["CD19", "CD3", "CD11c"])
        idx_interest = ff.get_indice(channels_of_interest)
        subsampleEmb2 = TSNE(n_components=2).fit_transform(subsample[:, idx_interest])
        df_subsampleEmb2 = pd.DataFrame(subsampleEmb2, columns = ['tsne1', 'tsne2'])
        df_result2 = pd.concat([df_subsampleEmb2, df_subsample], axis=1, sort=False)
        df_result2.to_csv('Resultats/2_Dimensionality_Reduction/2_result2_Dimension_Reduction.csv', index=False)

        plt.scatter(subsampleEmb2[:, 0], subsampleEmb2[:, 1])
        plt.title("tsne data of 3 channels CD3, CD11c and CD19 on 1000 cells")
        plt.xlabel('tsne1')
        plt.ylabel('tsne2')
        plt.savefig("Resultats/2_Dimensionality_Reduction/Figure_3 - tsne 1000 cells of channels of_interest.png")
        plt.show()
        

def main():
    _Dimensionality_Reduction('../inst/extdata/21-10-15_Tube_028_Live.fcs')

if __name__ == '__main__':
    main()