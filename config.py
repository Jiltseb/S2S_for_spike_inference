# Learning parameters
learning = {'rate' : 0.001,
            'minEpoch' : 2,
            'lrScale' : 0.9,
            'batchSize' : 16, #256,
            'lrScaleCount' : 1000,
            'minValError' : 0.00005}

# Feature extraction parameters: not used for spikefinder evaluations
param = {'windowLength':1000,'windowShift':1000,
         'fs': 100}
param['stdFloor'] = 1e-3  # Floor on standard deviation
param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0) #for 
param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

#main parameters
dataloc = '../../'  #train and test split location
maxlen = 100000     #max possible length of the calcium sigbnal to be fed to the model (100Hz, in samples)
