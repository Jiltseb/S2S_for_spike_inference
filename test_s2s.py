import numpy as np
from scipy.stats import spearmanr
from scipy import corrcoef
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from datasets import load_data_test
from optparse import OptionParser

#testing script for spikefinder

def score(a, b, method, downsample=4):
    """
    Estimate similarity score between two reslts.
    """
    methods = {
      'loglik': _loglik,
      'info': _info,
      'corr': _corr,
      'auc': _auc,
      'rank': _rank
    }
    if method not in methods.keys():
      raise Exception('scoring method not one of: %s' % ' '.join(methods.keys()))

    func = methods[method]

    result = [] 
    for i in range(a.shape[0]):
        x = a[i,:]
        y = b[i,:]
        x = x[:len(spike_npt[k])]
        ml = min([len(x),len(y)])

        x = x[0:ml]
        y = y[0:ml]
        naninds = np.isnan(x) | np.isnan(y)
        x = x[~naninds]
        y = y[~naninds]
        x = _downsample(x, downsample)
        y = _downsample(y, downsample)

        ml = min([len(x),len(y)])

        x = x[0:ml]
        y = y[0:ml]

        if not len(x) == len(y):
           raise Exception('mismatched lengths %s and %s' % (len(x), len(y)))

        if func=='info':
            result.append(func(x, y,fps=100/downsample))
        else:
            result.append(func(x, y))

    return result

def _corr(x, y):
    return corrcoef(x, y)[0,1]

def _rank(x, y):
    return spearmanr(x, y).correlation

def _auc(x, y):
     fpr, tpr, thresholds = roc_curve(y>0,x)
     return auc(fpr,tpr)

def _downsample(signal, factor):
    """
    Downsample signal by averaging neighboring values.
    @type  signal: array_like
    @param signal: one-dimensional signal to be downsampled
    @type  factor: int
    @param factor: this many neighboring values are averaged
    @rtype: ndarray
    @return: downsampled signal
    """

    if factor < 2:
        return np.asarray(signal)

    return np.convolve(np.asarray(signal).ravel(), np.ones(factor), 'valid')[::factor]






def model_test(model, test_dataset):
    #model.load_weights('model/model_conv_11_5')
    test_ip     = test_dataset['calcium signal padded']
    pred_test   = model.predict(test_ip)
    gt_test     = np.reshape(test_dataset['spikes train padded'],(test_ip.shape[0],-1))
    pred_test   = np.reshape(pred_test,(test_ip.shape[0],-1))
    corrs       = score(pred_test, gt_test, method='corr')
    corrs       = np.asarray(corrs)
    ranks       = score(pred_test, gt_test, method='rank')
    ranks       = np.asarray(ranks)
    aucs        = score(pred_test, gt_test, method='auc')
    aucs        = np.asarray(aucs)
    measures = []
    for i in range(5):
        corre = np.mean(corrs[id_staked_t==i])
        #print(corre)
        ranke = np.mean(ranks[id_staked_t==i])
        #print(ranke)
        auce = np.mean(aucs[id_staked_t==i])
        #print(auce)
        measures.append([corre, ranke, auce])
    return measures


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=1,keepdims=True)
    my = K.mean(y, axis=1,keepdims=True)
    xm, ym = x-mx, y-my 
    r_num = K.sum(xm*ym, axis=1)
    r_den = K.sqrt(K.sum(K.square(xm),axis=1) * K.sum(K.square(ym),axis=1))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

if __name__== '__main__':

    usage = 'USAGE: %prog model_path'
    parser = OptionParser(usage=usage)
    opts, args = parser.parse_args()

    if len(args) != 1:
        parser.usage += '\n\n' + parser.format_option_help()
        parser.error('Wrong number of arguments')
    
    model           = args[0] #model file location
    test_dataset    = load_data_test()
    id_staked_t     = test_dataset['ids stacked']
    spike_npt     = test_dataset['spikes train']

    m               = load_model (model, compile=False )
    results         = model_test (m, test_dataset)
    print(results)

