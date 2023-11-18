from time import time
import tensorflow as tf
from keras.models import Model
import keras.backend as K
from keras.engine.base_layer import InputSpec
from keras.engine.topology import Layer
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from layers import ConstantDispersionLayer, SliceLayer,ColWiseMultLayer
from sklearn.model_selection import train_test_split
from ZINBAutoEncoder import SCDeepCl
from loss import poisson_loss, NB, ZINB
from preprocess import read_dataset, normalize
import h5py
from normal import normalize_tr,normalize_te,log_sc
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2211)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')


def DeepZINB(X_train_,y_train,X_test_,y_test,batch_size):
        
        pretrain_epochs=800
        optimizer1 = Adam(amsgrad=True)
        w=0.2
        sf,train_raw,X_train,var_names,y_train=normalize_tr(X_train_,y_train)
        sf_test,test_raw,X_test,y_test=normalize_te(X_test_,var_names,y_test)
        print(train_raw.shape)
        print(test_raw.shape)
        batch_size=batch_size
        update_interval = int(X_train.shape[0]/batch_size)
        input_size=X_train.shape[1]
        gamma=1
        scDeepCl = SCDeepCl(dims=[input_size,64,32,16])
        ae_weight_file='ae_weights.h5'
        ae_weights='ae_weights.h5'
        save_dir='results/scDeepCl'
        scDeepCl.pretrain(x=[X_train, sf], y=train_raw, batch_size=batch_size, epochs=pretrain_epochs, optimizer=optimizer1, ae_file=ae_weight_file)
        pi,disp,meanz=scDeepCl.extract_parameter(x=[X_train, sf])
        pi_t,disp_t,mean_t=scDeepCl.extract_parameter(x=[X_test, sf_test])
        meanz=log_sc(meanz)
        mean=w*meanz+train_raw
        parameter={'pi': pi,  'disp' :disp,    'pi_t': pi_t,   'disp_t':disp_t,    'X_train':(mean),   'X_test':test_raw, 'y_train':y_train,   'y_test':y_test}
        K.clear_session()
        return   parameter
    





