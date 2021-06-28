# python script for loading spikefinder data
# for more info see https://github.com/codeneuro/spikefinder

import numpy as np
import pandas as pd
from scipy import signal
import config

def load_data_train():
    calcium_train = []
    spikes_train = []
    ids = []
    for dataset in range(10):
        calcium_train.append(np.array(pd.read_csv(config.dataloc + 
            'spikefinder.train/' + str(dataset+1) + 
            '.train.calcium.csv')))
        spikes_train.append(np.array(pd.read_csv(config.dataloc + 
            'spikefinder.train/' + str(dataset+1) + 
            '.train.spikes.csv')))
        ids.append(np.array([dataset]*calcium_train[-1].shape[1]))
    maxlen = max([c.shape[0] for c in calcium_train])
    maxlen = max(maxlen, config.maxlen)
    #maxlen_test = max([c.shape[0] for c in calcium_test])
    #maxlen = max(maxlen, maxlen_test)  
    calcium_train_padded = np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)), 'wrap' ) for c in calcium_train])
    spikes_train_padded = np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)), 'wrap' ) for c in spikes_train])
    ids_stacked = np.hstack(ids)
    sample_weight = 1. + 1.5*(ids_stacked<5)
    sample_weight /= sample_weight.mean()
    calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
    spikes_train_padded[np.isnan(spikes_train_padded)] = -1 #it was -1.

    calcium_train_padded[spikes_train_padded<-1] =  np.nan
    spikes_train_padded[spikes_train_padded<-1] =  np.nan
    #if gaussian convolving is needed
    window = signal.gaussian(33,std=10)
    spikes_train_padd = spikes_train_padded
    for i in range(spikes_train_padded.shape[1]):
        spikes_train_padd[:,i] = np.convolve(spikes_train_padded[:,i], window, mode='same')

    sp = np.asarray(spikes_train)
    calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
    spikes_train_padded[np.isnan(spikes_train_padded)] = -1 #it was -1.
    spikes_train_padd[np.isnan(spikes_train_padd)] = -1


    calcium_train_padded = calcium_train_padded.T[:, :, np.newaxis]
    spikes_train_padded = spikes_train_padded.T[:, :, np.newaxis]
    spikes_train_padd = spikes_train_padd.T[:, :, np.newaxis] 

    #optional-used mainly in test set
    ids_oneshot = np.zeros((calcium_train_padded.shape[0],
        calcium_train_padded.shape[1], 10))

    for n,i in enumerate(ids_stacked):
        ids_oneshot[n, :, i] = 1.

    return {'calcium signal padded': calcium_train_padded, 'spikes train padded': spikes_train_padded, 'Gaussian spikes train': spikes_train_padd}
    #optional to use either spike train or Gaussian train

def load_data_test():
    calcium_test = []
    spikes_test = []
    ids_test = []
    for dataset in range(5):
        calcium_test.append(np.array(pd.read_csv(config.dataloc +
                'spikefinder.test/' + str(dataset+1) +
                '.test.calcium.csv')))
        spikes_test.append(np.array(pd.read_csv(config.dataloc +
            'spikefinder.test/' + str(dataset+1) +
            '.test.spikes.csv')))
        ids_test.append(np.array([dataset]*calcium_test[-1].shape[1]))
    
    maxlen_test = max([c.shape[0] for c in calcium_test])
    maxlen_test = max(maxlen_test, config.maxlen)
    calcium_test_padded = \
        np.hstack([np.pad(c, ((0, maxlen_test-c.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for c in calcium_test])
    spikes_test_padded = \
        np.hstack([np.pad(c, ((0, maxlen_test-c.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for c in spikes_test])

    ids_test_stacked = np.hstack(ids_test)
    calcium_test_padded[spikes_test_padded<-1] =  np.nan
    spikes_test_padded[spikes_test_padded<-1] =  np.nan
    spt = np.asarray(spikes_test)

    calcium_test_padded[np.isnan(calcium_test_padded)] = 0.
    spikes_test_padded[np.isnan(spikes_test_padded)] = -1 

    calcium_test_padded = calcium_test_padded.T[:, :, np.newaxis]
    spikes_test_padded = spikes_test_padded.T[:, :, np.newaxis]

    ids_oneshot_test = np.zeros((calcium_test_padded.shape[0],
        calcium_test_padded.shape[1], 10))
    for n,i in enumerate(ids_test_stacked):
        ids_oneshot_test[n, :, i] = 1.

    return {'calcium signal': calcium_test, 'calcium signal padded': calcium_test_padded, 'spikes train': spt, 'spikes train padded': spikes_test_padded, 'ids oneshot': ids_oneshot_test, 'ids stacked': ids_test_stacked}      
