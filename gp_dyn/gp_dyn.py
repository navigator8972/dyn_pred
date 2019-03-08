'''
Gaussian Process Dynamics based on gpflow
'''
from __future__ import print_function
import tensorflow as tf
import gpflow

import numpy as np
import scipy

class GaussianProcessDynamics(object):
    def __init__(self, kern=gpflow.kernels.RBF(1)):
        self.gp_model = None
        self.kernel = kern

        #parameters for unscented kf like propagation with variance
        #see unscented kf paper for more details
        self.alpha=1e-3
        self.kappa=0
        self.beta=2.
        return
    
    def build_model(self, model, inputs, outputs):
        #assign a model but keep it untouched
        self.gp_model = model(inputs, outputs, self.kernel)
        self.input_dim = inputs.shape[1]
        self.output_dim = outputs.shape[1]
        return

    def train(self, inputs, outputs):
        if self.gp_model is None:
            self.build_model(gpflow.models.GPR, inputs, outputs)
        
        self.gp_model.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.gp_model)
        return
    
    def predict_f(self, inputs):
        assert(self.gp_model is not None)
        return self.gp_model.predict_f(inputs)
    
