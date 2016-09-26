import tensorflow as tf
import numpy as numpy
import numpy.random as npr
import random

class TFModel(object):
    def __init__(self, session, **hyperparams):
        print "Model Loading...",
        self.hyperparams = hyperparams
        for var, val in hyperparams.items():
            setattr(self, var, val)
        
        self.manipulate_params()
        self.build_forward()
        self.build_loss()
        self.build_optimizer()
        
        self.session = session
        self.session.run(tf.initialize_all_variables())
        print "Done"
        
    def manipulate_params(self):
        pass
    
    def build_forward(self):
        raise NotImplementedError, "Must implement the forward graph"
        
    def build_loss(self):
        raise NotImplementedError, "Must implement a loss function"
    
    def build_optimizer(self):
        raise NotImplementedError, "Must implement an optimizer"
        
    def batch_generator(self, x, y, batch_size=64):
        # shuffle the data
        xy = zip(x,y)
        random.shuffle(xy)
        x, y = zip(*xy)
        
        # create minibatches
        num_batches = len(y) // batch_size + 1
        for batch_i in range(num_batches):
            start, end = batch_i*batch_size, (batch_i+1)*batch_size
            yield batch_i, x[start:end], y[start:end]
    
    def print_progress(self, epoch_i, n_epoch, batch_i, n_batch, loss):
        if batch_i == 0:
            print
        print "\r Epoch {0}/{1} : Batch {2}/{3}: Loss {4:.4f}".format(
              epoch_i+1, n_epoch, batch_i+1, n_batch, loss),
        
    def fit(self, x, y, n_epoch=1, batch_size=64, **partial_fit_kwargs):
        n_batch = len(y) // batch_size + 1
        for epoch_i in range(n_epoch):
            for batch_i, batch_x, batch_y in self.batch_generator(x, y, batch_size): 
                loss = self.partial_fit(batch_x, batch_y)
                self.print_progress(epoch_i, n_epoch, batch_i, n_batch, loss)
                
    def partial_fit(self, x, y, measure_only=False):
        raise NotImplementedError, "Must implement a mini-batch fit"
    
    def predict(self, x):
        raise NotImplementedError, "Must implement a mini-batch prediction"
    
    def save(self):
        pass
    
    def load(self):
        pass