#!/usr/bin/python3.2
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import flowio
import flowutils

class Loadfcs():
    def __init__(self, file_address):
        fd = flowio.FlowData(file_address)
        data = np.reshape(fd.events, (-1, fd.channel_count))
        print(data.shape)
        '''
            - Number of cells
                print(fd.event_count)
            - Number of channel
                print(fd.channel_count)
            - dict of channels, info on keys(indice), value( PnN for channels and PnS for markers)
                print(fd.channels)
            - meta information on analysis and acquisition of data
                print(fd.text)
        '''    
        self.fd = fd
        self.meta = fd.text
        self.data = data

        self.meta_channels_markers = self.load_channels_markers_indices()

    def load_channels_markers_indices(self):
        list_indices = sorted({int(k) for k in self.fd.channels.keys()})
        dict_channels = {}
        dict_markers = {}
        for idx in list_indices:
            if 'PnN' in self.fd.channels[str(idx)] and 'PnS' in self.fd.channels[str(idx)]:
                dict_channels[self.fd.channels[str(idx)]['PnN']] = self.fd.channels[str(idx)]['PnS']
                dict_markers[self.fd.channels[str(idx)]['PnS']] = self.fd.channels[str(idx)]['PnN']
            else:
                dict_channels[self.fd.channels[str(idx)]['PnN']] = self.fd.channels[str(idx)]['PnN'] 
        meta_fd = np.array([list_indices, dict_channels, dict_markers])
        return meta_fd
        
    def get_all_indices(self):
        list_indices = self.meta_channels_markers[0]
        return list_indices

    def get_indice(self, channels):
        indices_channels = []
        for channel in self.fd.channels:
            if self.fd.channels[channel]['PnN'] in channels:
                    indices_channels.append(int(channel) - 1)
        return indices_channels

    def get_channels_by_indice(self, indices):
        channels =  []
        if isinstance(indices, list):    
            if all(isinstance(n, int) for n in indices): 
                for idx in indices:
                    channels.append(self.fd.channels[str(idx)]['PnN']) 
        elif isinstance(indices, int):    
                channels.append(self.fd.channels[str(indices)]['PnN']) 
        return channels

    def get_channels(self, markers):
        channels = []
        dict_markers = self.meta_channels_markers[2]
        if isinstance(markers, list):    
            for marker in markers:
                if marker in dict_markers:
                    channels.append(dict_markers[marker])
                else:
                    channels.append(marker)
        elif isinstance(markers, str):
            if marker in dict_markers:
                channels.append(dict_markers[markers])
            else:
                channels.append(markers)
        return channels

    def get_markers(self, channels):
        markers = []
        dict_channels = self.meta_channels_markers[1]
        if isinstance(channels, list): 
            for channel in channels:
                if channel in dict_channels.keys():
                    markers.append(dict_channels[channel])
                else:
                    markers.append(channel)
        elif isinstance(channels, str):
            if channels in dict_channels:
                markers.append(dict_channels[channels])
            else:
                markers.append(channels)
        return markers

    def compensate(self):
        #print(self.fd.text['spillover'])
        spill, spill_channels = flowutils.compensate.get_spill(self.fd.text['spillover']) 

        indices_channels = Loadfcs.get_indice(self, spill_channels)
        data_select = self.data[:, indices_channels]
        comp_result = np.linalg.solve(spill.T, data_select.T).T
        # Vérification de la solution
        verif = np.allclose(np.dot(spill.T,  comp_result.T), data_select.T)
        print(verif)

        data_comp = self.data.copy()
        data_comp[:, indices_channels] = comp_result

        return data_comp

    def transform(self, data, indices_channels, type_transform=None):
        '''type_transform can be 
            -hlog : hyperlog transformation
            -logicle : logicle transformation
        
        These values also need to be transformed.
        The  transformation takes several parameters, we provide the indices of 
        the columns that we want to transform:
        '''
        if type_transform == 'logicle':
            t_data = flowutils.transforms.logicle(data, indices_channels)
        elif type_transform == 'hlog':
            t_data = flowutils.transforms.hyperlog(data, indices_channels)
        else:
            t_data = data
        
        return t_data

    