'''
A wrapper of BNN for a consistent interface with variance propagation
'''

from __future__ import print_function
import tensorflow as tf

import numpy as np
import scipy

# from . DotmapUtils import *
from bnn_dyn.FC import FC
from bnn_dyn.BNN import BNN

from dotmap import DotMap

class BNNDynamics(object):
    def __init__(self, layers, n_nets=5, name="bnn_model", sess=None, output_activation=None):
        self.n_ensemble = n_nets

        if isinstance(layers, str):
            model_dir = layers
            load_model = True
        else:
            load_model = False
            model_dir = None

        params = DotMap(
            name=name, num_networks=n_nets,
            sess=sess, load_model=load_model,
            model_dir=model_dir
        )
        self.bnn_model = BNN(params)

        #construct layers if not loaded
        if not load_model:
            #the layers indicate dimensions of each layers
            self.input_dim = layers[0]
            for i in range(len(layers)-2):
                if i == 0:
                    self.bnn_model.add(FC(layers[1], input_dim=layers[0], activation="swish", weight_decay=0.000025))
                else:
                    self.bnn_model.add(FC(layers[i+1], activation="swish", weight_decay=0.000075))
            #output, linear output
            self.bnn_model.add(FC(layers[-1], activation=output_activation, weight_decay=0.0001))
            self.bnn_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        else:
            self.input_dim = self.bnn_model.layers[0].input_dim

        #put training parameters here for further tunning
        self.batch_size=32
        self.epochs=100
        self.hide_progress=False
        self.holdout_ratio=0.0
        self.max_logging=5000
        return
    
    def train(self, inputs, outputs):
        self.bnn_model.train(inputs, outputs, self.batch_size, self.epochs, 
            self.hide_progress, self.holdout_ratio, self.max_logging)

        return
    
    def predict_f(self, inputs):
        return self.bnn_model.predict(inputs, factored=False)

    
