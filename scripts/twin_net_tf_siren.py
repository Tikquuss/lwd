"""
try:
    %tensorflow_version 1.x
    %matplotlib inline
except Exception:
    pass
"""

# import and test
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())

# we want TF 1.x
assert tf.__version__ < "2.0"

# disable annoying warnings
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# import other useful libs
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
import random

from scipy.stats import norm

# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32
# real_type = tf.float64

from tensorflow.keras.initializers import he_uniform as bias_initializer, RandomUniform

from twin_net_tf import train, normalize_data  

## Feedforward neural network in TensorFlow
def vanilla_net(
    input_dim,      # dimension of inputs, e.g. 10
    hidden_units,   # units in hidden layers, assumed constant, e.g. 20
    hidden_layers,  # number of hidden layers, e.g. 4
    seed,           # seed for initialization or None for random
    first_omega_0, 
    hidden_omega_0,
    outermost_linear):          
    
    # set seed
    tf.set_random_seed(seed)
    
    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    
    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    omega_0 = [None]
    # layer 0 (input) has no parameters
    
    # layer 0 = input layer
    zs = [xs] # eq.3, l=0
    
    # first hidden layer (index 1)
    # weight matrix
    a = 1 / input_dim
    weight_initializer = RandomUniform.from_config(RandomUniform(-a, a).get_config())
    ws.append(tf.get_variable("w1", [input_dim, hidden_units], \
        initializer = weight_initializer, dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hidden_units], \
        initializer = bias_initializer(), dtype=real_type))
    # graph
    z = first_omega_0 * ws[1]
    zs.append(zs[0] @ z + bs[1]) # eq. 3, l=1
    omega_0.append(first_omega_0)
    
    # second hidden layer (index 2) to last (index hidden_layers) 
    a = np.sqrt(6 / hidden_units) / hidden_omega_0
    weight_initializer = RandomUniform.from_config(RandomUniform(-a, a).get_config())
    for l in range(1, hidden_layers): 
        ws.append(tf.get_variable("w%d"%(l+1), [hidden_units, hidden_units], \
            initializer = weight_initializer, dtype=real_type))
        bs.append(tf.get_variable("b%d"%(l+1), [hidden_units], \
            initializer = bias_initializer(), dtype=real_type))
        z = hidden_omega_0*ws[l+1] 
        zs.append(tf.math.sin(zs[l]) @ z + bs[l+1]) # eq. 3, l=2..L-1

        omega_0.append(hidden_omega_0)
        
    # output layer (index hidden_layers+1)
    ws.append(tf.get_variable("w"+str(hidden_layers+1), [hidden_units, 1], \
            initializer = weight_initializer, dtype=real_type))
    bs.append(tf.get_variable("b"+str(hidden_layers+1), [1], \
        initializer = bias_initializer(), dtype=real_type))
    # eq. 3, l=L
    z = hidden_omega_0*ws[hidden_layers+1] 
    if outermost_linear :
        zs.append(tf.math.sin(zs[hidden_layers]) @ z + bs[hidden_layers+1])
    else :
        zs.append(tf.math.sin(tf.math.sin(zs[hidden_layers]) @ z + bs[hidden_layers+1]))
    omega_0.append(hidden_omega_0)

    # result = output layer
    ys = zs[hidden_layers+1]
    
    # return input layer, (parameters = weight matrices and bias vectors), 
    # [all layers] and output layer
    return xs, (ws, bs, omega_0, outermost_linear), zs, ys
    
## Explicit backpropagation and twin network

# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
    weights_and_biases, # 2nd output from vanilla_net() 
    zs):                # 3rd output from vanilla_net()
    
    ws, bs, omega_0, outermost_linear = weights_and_biases
    L = len(zs) - 1
    
    # backpropagation, eq. 4, l=L..1
    if outermost_linear :
        zbar = tf.ones_like(zs[L]) # zbar_L = 1
    else :
        zbar = tf.math.cos(zs[L]) # zbar_L = 1

    for l in range(L-1, 0, -1):
        zbar = (zbar @ tf.transpose(omega_0[l+1] * ws[l+1])) * tf.math.cos(zs[l]) # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(omega_0[1] * ws[1]) # eq. 4
    
    xbar = zbar # xbar = zbar_0
    
    # dz[L] / dx
    return xbar    

# combined graph for valuation and differentiation
def twin_net(input_dim, hidden_units, hidden_layers, seed, first_omega_0, hidden_omega_0, outermost_linear):
    
    # first, build the feedforward net
    xs, (ws, bs, omega_0, outermost_linear), zs, ys = vanilla_net(input_dim, hidden_units, hidden_layers, seed, first_omega_0, hidden_omega_0, outermost_linear)
    
    # then, build its differentiation by backprop
    xbar = backprop((ws, bs, omega_0, outermost_linear), zs)
    
    # return input x, output y and differentials d_y/d_z
    return xs, ys, xbar
    
## Vanilla training loop
def vanilla_training_graph(input_dim, hidden_units, hidden_layers, seed, first_omega_0, hidden_omega_0, outermost_linear):
    
    # net
    inputs, weights_and_biases, layers, predictions = \
        vanilla_net(input_dim, hidden_units, hidden_layers, seed, first_omega_0, hidden_omega_0, outermost_linear)
    
    # backprop even though we are not USING differentials for training
    # we still need them to predict derivatives dy_dx 
    derivs_predictions = backprop(weights_and_biases, layers)
    
    # placeholder for labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    
    # loss 
    loss = tf.losses.mean_squared_error(labels, predictions)
    
    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary 
    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)

# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            inputs, labels, lr_placeholder, minimizer,   
                            # training set 
                            x_train, y_train,                           
                            # params, left to client code
                            learning_rate, batch_size, session):        
    
    m, n = x_train.shape
    
    # minimization loop over mini-batches
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)
        
## Differential training loop
def diff_training_graph(
    # same as vanilla
    input_dim, 
    hidden_units, 
    hidden_layers, 
    seed, 
    # balance relative weight of values and differentials 
    # loss = alpha * MSE(values) + beta * MSE(greeks, lambda_j) 
    # see online appendix
    alpha, 
    beta,
    lambda_j,
    first_omega_0, 
    hidden_omega_0, 
    outermost_linear):
    
    # net, now a twin
    inputs, predictions, derivs_predictions = twin_net(input_dim, hidden_units, hidden_layers, seed, first_omega_0, hidden_omega_0, outermost_linear)
    
    # placeholder for labels, now also derivs labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    derivs_labels = tf.placeholder(shape=[None, derivs_predictions.shape[1]], dtype=real_type)
    
    # loss, now combined values + derivatives
    loss = alpha * tf.losses.mean_squared_error(labels, predictions) \
    + beta * tf. losses.mean_squared_error(derivs_labels * lambda_j, derivs_predictions * lambda_j)
    
    # optimizer, as vanilla
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary tensors, including derivatives
    # predictions and labels
    return inputs, labels, derivs_labels, predictions, derivs_predictions, \
            learning_rate, loss, optimizer.minimize(loss)

def diff_train_one_epoch(inputs, labels, derivs_labels, 
                         # graph
                         lr_placeholder, minimizer,             
                         # training set, extended
                         x_train, y_train, dydx_train,          
                         # params
                         learning_rate, batch_size, session):   
    
    m, n = x_train.shape
    
    # minimization loop, now with Greeks
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            derivs_labels: dydx_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)
        

class Neural_Approximator():
    
    def __init__(self, x_raw, y_raw, dydx_raw=None, normalize = True):      # derivatives labels, 
       
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        self.normalize = normalize
        
        # tensorflow logic
        self.graph = None
        self.session = None
                        
        self.stats = {}
        
    def __del__(self):
        if self.session is not None:
            self.session.close()
        
    def build_graph(self,
                differential,       # differential or not           
                lam,                # balance cost between values and derivs  
                hidden_units, 
                hidden_layers, 
                weight_seed,
                first_omega_0, 
                hidden_omega_0, 
                outermost_linear):
        
        # first, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        self.graph = tf.Graph()
        
        with self.graph.as_default():
        
            # build the graph, either vanilla or differential
            self.differential = differential
            
            if not differential:
            # vanilla 
                
                self.inputs, \
                self.labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer \
                = vanilla_training_graph(self.n, hidden_units, hidden_layers, weight_seed, first_omega_0, hidden_omega_0, outermost_linear)
                    
            else:
            # differential
            
                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")
            
                self.alpha = 1.0 / (1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha
                
                self.inputs, \
                self.labels, \
                self.derivs_labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer = diff_training_graph(self.n, hidden_units, \
                                                     hidden_layers, weight_seed, \
                                                     self.alpha, self.beta, self.lambda_j, first_omega_0, hidden_omega_0, outermost_linear)
        
            # global initializer
            self.initializer = tf.global_variables_initializer()
            
        # done
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
                        
    # prepare for training with m examples, standard or differential
    def prepare(self, 
                m, 
                differential,
                lam=1,              # balance cost between values and derivs  
                # standard architecture
                hidden_units=20, 
                hidden_layers=4, 
                weight_seed=None,
                first_omega_0 = 30., 
                hidden_omega_0 = 30., 
                outermost_linear = True):

        # prepare dataset
        if self.normalize : 
            self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
                normalize_data(self.x_raw, self.y_raw, self.dydx_raw, m)
        else :

            self.x_mean, self.x_std, self.x = np.zeros_like(self.x_raw[0]), np.ones_like(self.x_raw[0]), self.x_raw
            self.y_mean, self.y_std, self.y = np.zeros_like(self.y_raw[0]), np.ones_like(self.y_raw[0]), self.y_raw
            self.dy_dx, self.lambda_j = self.dydx_raw, 1.

        # build graph        
        self.m, self.n = self.x.shape        
        self.build_graph(differential, lam, hidden_units, hidden_layers, weight_seed, first_omega_0, hidden_omega_0, outermost_linear)
        
    def train(self,            
              description="training",
              # training params
              reinit=True, 
              epochs=100, 
              # one-cycle learning rate schedule
              learning_rate_schedule=[
                  (0.0, 1.0e-8), 
                  (0.2, 0.1), 
                  (0.6, 0.01), 
                  (0.9, 1.0e-6), 
                  (1.0, 1.0e-8)], 
              batches_per_epoch=16,
              min_batch_size=256,
              # callback and when to call it
              # we don't use callbacks, but this is very useful, e.g. for debugging
              callback=None,           # arbitrary callable
              callback_epochs=[],     # call after what epochs, e.g. [5, 20]
              improving_limit =  float("inf")):

        self.stats['differential' if self.differential else "normal"] = train(description, 
              self, 
              reinit, 
              epochs, 
              learning_rate_schedule, 
              batches_per_epoch, 
              min_batch_size,
              callback, 
              callback_epochs,
              improving_limit)
     
    def predict_values(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std 
        # predict scaled
        y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        return y

    def predict_values_scaled(self, x_scaled):
        y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled})
        return y_scaled

    def predict_values_and_derivs(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std
        # predict scaled
        y_scaled, dyscaled_dxscaled = self.session.run(
            [self.predictions, self.derivs_predictions], 
            feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        dydx = self.y_std / self.x_std * dyscaled_dxscaled
        return y, dydx

    def predict_values_and_derivs_scaled(self, x_scaled):
        y_scaled, dyscaled_dxscaled = self.session.run(
            [self.predictions, self.derivs_predictions], 
            feed_dict = {self.inputs: x_scaled})
        return y_scaled, dyscaled_dxscaled
        
        
def test(generator, 
         sizes,
         nTrain,
         nTest, 
         trainSeed=None, 
         testSeed=None, 
         weightSeed=None, 
         deltidx=0, 
         generator_kwargs = {},
         epochs=100,
         first_omega_0 = 30, 
         hidden_omega_0 = 30., 
         outermost_linear = True,
         normalize = True,
         improving_limit = float("inf"),
         min_batch_size = 256):

    # simulation
    print("simulating training, valid and test sets")
    try :
        xTrain, yTrain, dydxTrain = generator.trainingSet(num_samples = nTrain, seed = trainSeed)
        xTest, yTest, dydxTest = generator.testSet(num_samples = nTest, seed = testSeed)
        xAxis = np.array([None])
        vegas = np.array([None])
    except (ValueError, TypeError) : # too many values to unpack (expected 2), trainingSet() got an unexpected keyword argument 'num_samples'
        xTrain, yTrain, dydxTrain = generator.trainingSet(nTrain, seed = trainSeed)
        xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num = nTest, seed = testSeed)

    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain, normalize = normalize)
    print("done")
    
    predvalues = {}    
    preddeltas = {}

    loss_function = tf.losses.mean_squared_error
    dic_loss = {}
    dic_loss['standard_loss'], dic_loss['differential_loss'] = {}, {}
    dic_loss['standard_loss']["yloss"], dic_loss['standard_loss']["dyloss"] = [], []
    dic_loss['differential_loss']["yloss"], dic_loss['differential_loss']["dyloss"] = [], []

    for size in sizes:        
            
        print("\nsize %d" % size)
        regressor.prepare(m = size, differential= False, weight_seed = weightSeed, **generator_kwargs, 
                         first_omega_0 = first_omega_0, hidden_omega_0 = hidden_omega_0, outermost_linear = outermost_linear)
            
        t0 = time.time()
        regressor.train("standard training", epochs = epochs, improving_limit = improving_limit, min_batch_size = min_batch_size)
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()
        
        with tf.Session() as sess:
            loss = sess.run([loss_function(yTest, predictions), loss_function(dydxTest, deltas)])
            print('test y loss : {}, test dy loss : {}'.format(loss[0], loss[1]))
            dic_loss['standard_loss']["yloss"].append(loss[0])
            dic_loss['standard_loss']["dyloss"].append(loss[1])

        regressor.prepare(m = size, differential = True, weight_seed = weightSeed, **generator_kwargs, 
                         first_omega_0 = first_omega_0, hidden_omega_0 = hidden_omega_0, outermost_linear = outermost_linear)
            
        t0 = time.time()
        regressor.train("differential training", epochs = epochs, improving_limit = improving_limit, min_batch_size = min_batch_size)
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()
        
        with tf.Session() as sess:
            loss = sess.run([loss_function(yTest, predictions), loss_function(dydxTest, deltas)])
            print('test y loss : {}, test dy loss : {}'.format(loss[0], loss[1]))
            dic_loss['differential_loss']["yloss"].append(loss[0])
            dic_loss['differential_loss']["dyloss"].append(loss[1])
    
    if xAxis.all() :
        return dic_loss, regressor, (xTrain, yTrain, dydxTrain), (xTest, yTest, dydxTest), dydxTest[:, deltidx], predvalues, preddeltas, xAxis, vegas
    else :
        return dic_loss, regressor, (xTrain, yTrain, dydxTrain), (xTest, yTest, dydxTest), dydxTest[:, deltidx], predvalues, preddeltas
     
