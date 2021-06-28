from optparse import OptionParser
import keras
import numpy
from datasets import load_data_train
from keras import backend as K
import os
# training script of s2s

usage = 'USAGE: %python train_s2s.py model_outdir'

parser = OptionParser(usage=usage)
opts, args = parser.parse_args()

if len(args) != 1:
    parser.usage += '\n\n' + parser.format_option_help()
    parser.error('Wrong number of arguments')

model_dir = args[0]

train_dataset = load_data_train()
train_ip = train_dataset['calcium signal padded']
train_op = train_dataset['Gaussian spikes train']

inputFeatDim = train_ip.shape[1]

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

# Initialise learning parameters and models
s = keras.optimizers.Adam(lr=config.learning['rate'], decay=0)

# Model definition
numpy.random.seed(25)
m = keras.models.Sequential()
m.add(keras.layers.Reshape((100000, 1), input_shape=(100000,1)))
m.add(keras.layers.Conv1D(filters=30, kernel_size=100, strides=1,  padding='same', use_bias=False))# kernel_constraint=non_neg()
m.add(keras.layers.Activation('relu'))

m.add(keras.layers.TimeDistributed(keras.layers.Dense(30), input_shape=(m.output_shape[1], m.output_shape[2])))
m.add(keras.layers.Activation('relu'))
m.add(keras.layers.Dropout(0.2))

m.add(keras.layers.TimeDistributed(keras.layers.Dense(30)))
m.add(keras.layers.Activation('relu'))
m.add(keras.layers.Dropout(0.2))

m.add(keras.layers.TimeDistributed(keras.layers.Dense(30)))
m.add(keras.layers.Activation('relu'))
m.add(keras.layers.Dropout(0.2))

m.add(keras.layers.core.Lambda (lambda x:K.expand_dims(x, axis=2)))
m.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=(100,1), strides=(1,1), padding='same', use_bias = False))
m.add(keras.layers.core.Lambda (lambda x:K.squeeze(x, axis=3)))
m.add(keras.layers.core.Lambda (lambda x:x[:,:100000,:]))
m.summary()    


# training
m.compile(loss=correlation_coefficient_loss, optimizer=s, metrics=['mse'])
r = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config.learning['lrScale'], patience=4,
        verbose=1, min_delta=config.learning['minValError'], cooldown=1, min_lr=config.learning['rate']) 
e = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)

h = [m.fit(train_ip, train_op, batch_size=20, epochs=100, verbose=2, validation_split=0.2, callbacks=[r,e])]
m.save(os.path.join(model_dir,'model_s2s.h5'), overwrite=True)
