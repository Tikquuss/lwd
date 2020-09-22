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
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


#from utils import genData
from utils import genData

# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32
# real_type = tf.float64


## Feedforward neural network in TensorFlow
def vanilla_net(
    input_dim,      # dimension of inputs, e.g. 10
    hidden_units,   # units in hidden layers, assumed constant, e.g. 20
    hidden_layers,  # number of hidden layers, e.g. 4
    seed):          # seed for initialization or None for random
    
    # set seed
    tf.set_random_seed(seed)
    
    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    
    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    # layer 0 (input) has no parameters
    
    # layer 0 = input layer
    zs = [xs] # eq.3, l=0
    
    # first hidden layer (index 1)
    # weight matrix
    ws.append(tf.get_variable("w1", [input_dim, hidden_units], \
        initializer = tf.variance_scaling_initializer(), dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hidden_units], \
        initializer = tf.zeros_initializer(), dtype=real_type))
    # graph
    zs.append(zs[0] @ ws[1] + bs[1]) # eq. 3, l=1
    
    # second hidden layer (index 2) to last (index hidden_layers)
    for l in range(1, hidden_layers): 
        ws.append(tf.get_variable("w%d"%(l+1), [hidden_units, hidden_units], \
            initializer = tf.variance_scaling_initializer(), dtype=real_type))
        bs.append(tf.get_variable("b%d"%(l+1), [hidden_units], \
            initializer = tf.zeros_initializer(), dtype=real_type))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l+1] + bs[l+1]) # eq. 3, l=2..L-1

    # output layer (index hidden_layers+1)
    ws.append(tf.get_variable("w"+str(hidden_layers+1), [hidden_units, 1], \
            initializer = tf.variance_scaling_initializer(), dtype=real_type))
    bs.append(tf.get_variable("b"+str(hidden_layers+1), [1], \
        initializer = tf.zeros_initializer(), dtype=real_type))
    # eq. 3, l=L
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers+1] + bs[hidden_layers+1]) 
    
    # result = output layer
    ys = zs[hidden_layers+1]
    
    # return input layer, (parameters = weight matrices and bias vectors), 
    # [all layers] and output layer
    return xs, (ws, bs), zs, ys
    
## Explicit backpropagation and twin network

# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
    weights_and_biases, # 2nd output from vanilla_net() 
    zs):                # 3rd output from vanilla_net()
    
    ws, bs = weights_and_biases
    L = len(zs) - 1
    
    # backpropagation, eq. 4, l=L..1
    zbar = tf.ones_like(zs[L]) # zbar_L = 1
    for l in range(L-1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l]) # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(ws[1]) # eq. 4
    
    xbar = zbar # xbar = zbar_0
    
    # dz[L] / dx
    return xbar    

# combined graph for valuation and differentiation
def twin_net(input_dim, hidden_units, hidden_layers, seed):
    
    # first, build the feedforward net
    xs, (ws, bs), zs, ys = vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
    # then, build its differentiation by backprop
    xbar = backprop((ws, bs), zs)
    
    # return input x, output y and differentials d_y/d_z
    return xs, ys, xbar
    
## Vanilla training loop
def vanilla_training_graph(input_dim, hidden_units, hidden_layers, seed):
    
    # net
    inputs, weights_and_biases, layers, predictions = \
        vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
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
    lambda_j):
    
    # net, now a twin
    inputs, predictions, derivs_predictions = twin_net(input_dim, hidden_units, hidden_layers, seed)
    
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
        
## Combined outer training loop

def train(description,
          # neural approximator
          approximator,              
          # training params
          reinit=True, 
          epochs=100, 
          # one-cycle learning rate schedule
          learning_rate_schedule=[    (0.0, 1.0e-8), \
                                      (0.2, 0.1),    \
                                      (0.6, 0.01),   \
                                      (0.9, 1.0e-6), \
                                      (1.0, 1.0e-8)  ], 
          batches_per_epoch=16,
          min_batch_size=256,
          # callback function and when to call it
          callback=None,           # arbitrary callable
          callback_epochs=[],      # call after what epochs, e.g. [5, 20]
          improving_limit = 100):     
              
    # batching
    batch_size = max(min_batch_size, approximator.m // batches_per_epoch)
    
    # one-cycle learning rate sechedule
    lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)
            
    # reset
    if reinit:
        approximator.session.run(approximator.initializer)
    
    # callback on epoch 0, if requested
    if callback and 0 in callback_epochs:
        callback(approximator, 0)
    
    #import pickle
    #import copy
    #tmp_best_model_path = './best-model-tmp.h5'
    stats = {}
    stats['train_yloss'] = []
    stats['train_dyloss'] = []
    best_loss = float('inf')
    counter = 1

    # loop on epochs, with progress bar (tqdm)
    for epoch in tqdm_notebook(range(epochs), desc=description):
        
        # interpolate learning rate in cycle
        learning_rate = np.interp(epoch / epochs, lr_schedule_epochs, lr_schedule_rates)
        
        # train one epoch
        
        if not approximator.differential:
        
            vanilla_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                learning_rate, 
                batch_size, 
                approximator.session)
        
        else:
        
            diff_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.derivs_labels,
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                approximator.dy_dx,
                learning_rate, 
                batch_size, 
                approximator.session)
        
        # callback, if requested
        if callback and epoch in callback_epochs:
            callback(approximator, epoch)

        with tf.Session() as sess:
            predictions, deltas = approximator.predict_values_and_derivs_scaled(approximator.x)
            
            loss = sess.run([
                             tf.losses.mean_squared_error(approximator.y, predictions), 
                             tf.losses.mean_squared_error(approximator.dy_dx, deltas)]
                            )
            print('Epoch {}: y loss scaled : {}, dy loss _scaled : {}'.format(epoch, loss[0], loss[1]))
            print()

            y = approximator.y*approximator.y_std + approximator.y_mean
            predictions = predictions*approximator.y_std + approximator.y_mean

            dydx = approximator.x_std / approximator.y_std * approximator.dy_dx
            deltas = approximator.x_std / approximator.y_std * deltas

            loss = sess.run([
                             tf.losses.mean_squared_error(y, predictions), 
                             tf.losses.mean_squared_error(dydx, deltas)]
                            )
            print('y loss no scaled : {}, dy loss no scaled : {}'.format(loss[0], loss[1]))
            print()
            print()

            stats['train_yloss'].append(loss[0])
            stats['train_dyloss'].append(loss[1])


            if loss[0] < best_loss :
                best_loss = loss[0]
                counter = 1
                #a = approximator.session
                #approximator.session = None
                #pickle.dump(approximator, open(tmp_best_model_path, 'wb'))
                #torch.save(approximator, tmp_best_model_path)
                #a = copy.deepcopy(approximator)
                #approximator.session = a
            else :
                counter += 1

            if counter == improving_limit + 1:
                break

    # final callback, if requested
    if callback and epochs in callback_epochs:
        callback(approximator, epochs)     

    #approximator = pickle.load(open(tmp_best_model_path, 'rb'))
    #approximator = torch.load(tmp_best_model_path)

    return stats

## Data normalization
# basic data preparation
epsilon = 1.0e-08
def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
    
    # crop dataset
    m = crop if crop is not None else x_raw.shape[0]
    x_cropped = x_raw[:m]
    y_cropped = y_raw[:m]
    dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
    
    # normalize dataset
    x_mean = x_cropped.mean(axis=0)
    x_std = x_cropped.std(axis=0) + epsilon
    x = (x_cropped- x_mean) / x_std
    y_mean = y_cropped.mean(axis=0)
    y_std = y_cropped.std(axis=0) + epsilon
    y = (y_cropped-y_mean) / y_std
    
    # normalize derivatives too
    if dycropped_dxcropped is not None:
        dy_dx = dycropped_dxcropped / y_std * x_std 
        # weights of derivatives in cost function = (quad) mean size
        lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
    else:
        dy_dx = None
        lambda_j = None
    
    return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j
    
class Neural_Approximator():
    
    def __init__(self, x_raw, y_raw, dydx_raw = None, normalize = True):      # derivatives labels, 
       
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
                weight_seed):
        
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
                = vanilla_training_graph(self.n, hidden_units, hidden_layers, weight_seed)
                    
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
                                                     self.alpha, self.beta, self.lambda_j)
                
        
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
                weight_seed=None):

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
        self.build_graph(differential, lam, hidden_units, hidden_layers, weight_seed)
        
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
              
        self.stats['differential' if self.differential else "normal"]  = train(
              description, 
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
        
    
# main class
class Generator :
    
    def __init__(self, callable_function, callable_function_deriv, dim_x, min_x, max_x):
        
        self.callable_function = callable_function
        self.callable_function_deriv = callable_function_deriv
        self.dim_x = dim_x
        self.min_x = min_x 
        self.max_x = max_x

    # training set: returns x, y and dx/dy
    def trainingSet(self, num_samples, seed = None):
      
        random.seed(seed)
        np.random.seed(seed)
        
        batch_samples = genData(
                        function = self.callable_function, 
                        deriv_function = self.callable_function_deriv, 
                        dim_x = self.dim_x,
                        min_x = self.min_x, max_x = self.max_x, num_samples = num_samples
                )

        X = np.array([bs[0] for bs in batch_samples])
        Y = np.array([[bs[1]] for bs in batch_samples])
        Z = np.array([bs[2] for bs in batch_samples])
        return X, Y, Z
   
    def testSet(self, num_samples, seed = None):

        random.seed(seed)
        np.random.seed(seed)
        
        batch_samples = genData(
                        function = self.callable_function, 
                        deriv_function = self.callable_function_deriv, 
                        dim_x = self.dim_x,
                        min_x = self.min_x, max_x = self.max_x, num_samples = num_samples
                )

        X = np.array([bs[0] for bs in batch_samples])
        Y = np.array([[bs[1]] for bs in batch_samples])
        Z = np.array([bs[2] for bs in batch_samples])
        
        return X, Y, Z
        
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
         normalize = True,
         improving_limit = float("inf")):

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
        regressor.prepare(m = size, differential= False, weight_seed = weightSeed, **generator_kwargs)
            
        t0 = time.time()
        regressor.train("standard training", epochs=epochs, improving_limit = improving_limit)
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()

        with tf.Session() as sess:
            loss = sess.run([loss_function(yTest, predictions), loss_function(dydxTest, deltas)])
            print('test y loss : {}, test dy loss : {}'.format(loss[0], loss[1]))
            dic_loss['standard_loss']["yloss"].append(loss[0])
            dic_loss['standard_loss']["dyloss"].append(loss[1])

        regressor.prepare(m = size, differential = True, weight_seed = weightSeed, **generator_kwargs)
            
        t0 = time.time()
        regressor.train("differential training", epochs=epochs, improving_limit = improving_limit)
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
     

def graph(title, 
          predictions, 
          xAxis, 
          xAxisName, 
          yAxisName, 
          targets, 
          sizes, 
          computeRmse=False, 
          weights=None):
    
    numRows = len(sizes)
    numCols = 2

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

    for i, size in enumerate(sizes):
        ax[i,0].annotate("size %d" % size, xy=(0, 0.5), 
          xytext=(-ax[i,0].yaxis.labelpad-5, 0),
          xycoords=ax[i,0].yaxis.label, textcoords='offset points',
          ha='right', va='center')
  
    ax[0,0].set_title("standard")
    ax[0,1].set_title("differential")
    
    for i, size in enumerate(sizes):        
        for j, regType, in enumerate(["standard", "differential"]):

            if computeRmse:
                errors = 100 * (predictions[(regType, size)] - targets)
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName
                
            ax[i,j].set_xlabel(t)            
            ax[i,j].set_ylabel(yAxisName)

            ax[i,j].plot(xAxis*100, predictions[(regType, size)]*100, 'co', \
                         markersize=2, markerfacecolor='white', label="predicted")
            ax[i,j].plot(xAxis*100, targets*100, 'r.', markersize=0.5, label='targets')

            ax[i,j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    plt.show()


## Black & Scholes
# helper analytics    
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)

def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)

def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)
#
    
# main class
class BlackScholes:
    
    def __init__(self, 
                 vol=0.2,
                 T1=1, 
                 T2=2, 
                 K=1.10,
                 volMult=1.5):
        
        self.spot = 1
        self.vol = vol
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult
                        
    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, anti=True, seed=None):
    
        np.random.seed(seed)
        
        # 2 sets of normal returns
        returns = np.random.normal(size=[m, 2])

        # SDE
        vol0 = self.vol * self.volMult
        R1 = np.exp(-0.5*vol0*vol0*self.T1 + vol0*np.sqrt(self.T1)*returns[:,0])
        R2 = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1) \
                    + self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
        S1 = self.spot * R1
        S2 = S1 * R2 

        # payoff
        pay = np.maximum(0, S2 - self.K)
        
        # two antithetic paths
        if anti:
            
            R2a = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1) \
                    - self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
            S2a = S1 * R2a             
            paya = np.maximum(0, S2a - self.K)
            
            X = S1
            Y = 0.5 * (pay + paya)
    
            # differentials
            Z1 =  np.where(S2 > self.K, R2, 0.0).reshape((-1,1)) 
            Z2 =  np.where(S2a > self.K, R2a, 0.0).reshape((-1,1)) 
            Z = 0.5 * (Z1 + Z2)
                    
        # standard
        else:
        
            X = S1
            Y = pay
            
            # differentials
            Z =  np.where(S2 > self.K, R2, 0.0).reshape((-1,1)) 
        
        return X.reshape([-1,1]), Y.reshape([-1,1]), Z.reshape([-1,1])
    
    # test set: returns a grid of uniform spots 
    # with corresponding ground true prices, deltas and vegas
    def testSet(self, lower=0.35, upper=1.65, num=100, seed=None):
        
        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, deltas and vegas
        prices = bsPrice(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bsDelta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas = bsVega(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices, deltas, vegas   

## Gaussian basket options
# helper analytics
def bachPrice(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return  vol * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))

def bachDelta(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return norm.cdf(d)

def bachVega(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return np.sqrt(T) * norm.pdf(d)
#
    
# generates a random correlation matrix
def genCorrel(n):
    randoms = np.random.uniform(low=-1., high=1., size=(2*n, n))
    cov = randoms.T @ randoms
    invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])

# main class
class Bachelier:
    
    def __init__(self, 
                 n,
                 T1=1, 
                 T2=2, 
                 K=1.10,
                 volMult=1):
        
        self.n = n
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult
                
    # training set: returns S1 (mxn), C2 (mx1) and dC2/dS1 (mxn)
    def trainingSet(self, m, anti=True, seed=None, bktVol=0.2):
    
        np.random.seed(seed)

        # spots all currently 1, without loss of generality
        self.S0 = np.repeat(1., self.n)
        # random correl
        self.corr = genCorrel(self.n)

        # random weights
        self.a = np.random.uniform(low=1., high=10., size=self.n)
        self.a /= np.sum(self.a)
        # random vols
        vols = np.random.uniform(low=5., high = 50., size = self.n)
        # normalize vols for a given volatility of basket, 
        # helps with charts without loss of generality
        avols = (self.a * vols).reshape((-1,1))
        v = np.sqrt(np.linalg.multi_dot([avols.T, self.corr, avols]).reshape(1))
        self.vols = vols * bktVol / v
        self.bktVol = bktVol

        # Choleski etc. for simulation
        diagv = np.diag(self.vols)
        self.cov = np.linalg.multi_dot([diagv, self.corr, diagv])
        self.chol = np.linalg.cholesky(self.cov) * np.sqrt(self.T2 - self.T1)
        # increase vols for simulation of X so we have more samples in the wings
        self.chol0 = self.chol * self.volMult * np.sqrt(self.T1 / (self.T2 - self.T1))
        # simulations
        normals = np.random.normal(size=[2, m, self.n])
        inc0 = normals[0, :, :] @ self.chol0.T
        inc1 = normals[1, :, :] @ self.chol.T
    
        S1 = self.S0 + inc0
        
        S2 = S1 + inc1
        bkt2 = np.dot(S2, self.a)
        pay = np.maximum(0, bkt2 - self.K)

        # two antithetic paths
        if anti:
            
            S2a = S1 - inc1
            bkt2a = np.dot(S2a, self.a)
            paya = np.maximum(0, bkt2a - self.K)
            
            X = S1
            Y = 0.5 * (pay + paya)
    
            # differentials
            Z1 =  np.where(bkt2 > self.K, 1.0, 0.0).reshape((-1,1)) * self.a.reshape((1,-1))
            Z2 =  np.where(bkt2a > self.K, 1.0, 0.0).reshape((-1,1)) * self.a.reshape((1,-1))
            Z = 0.5 * (Z1 + Z2)
                    
        # standard
        else:
        
            X = S1
            Y = pay
            
            # differentials
            Z =  np.where(bkt2 > self.K, 1.0, 0.0).reshape((-1,1)) * self.a.reshape((1,-1))
            
        return X, Y.reshape((-1,1)), Z
    
    # test set: returns an array of independent, uniformly random spots 
    # with corresponding baskets, ground true prices, deltas and vegas
    def testSet(self, lower=0.50, upper=1.50, num=4096, seed=None):
        np.random.seed(seed)
        spots = np.random.uniform(low=lower, high = upper, size=(num, self.n))
        baskets = np.dot(spots, self.a).reshape((-1, 1))
        prices = bachPrice(baskets, self.K, self.bktVol, self.T2 - self.T1)
        deltas = bachDelta(baskets, self.K, self.bktVol, self.T2 - self.T1) \
            @ self.a.reshape((1, -1))
        vegas = bachVega(baskets, self.K, self.bktVol, self.T2 - self.T1) 
        return spots, baskets, prices.reshape((-1, 1)), deltas, vegas

from utils import normalize_data as normalize_data_torch
def get_diffML_data_loader(generator, nTrain, nTest, train_seed, test_seed, batch_size = 32, with_derivative = False, normalize = False):
  
    xTrain, yTrain, dydxTrain = generator.trainingSet(nTrain, seed = train_seed)
    xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num = nTest, seed = test_seed)

    _, n = xTrain.shape
    if normalize :
        cond = not (dydxTrain is None)
        (x_mean, x_std, xTrain), (y_mean, y_std, yTrain), (dydx_mean, dydx_std, dydxTrain) = normalize_data_torch(
                                                                            xTrain, yTrain, dydxTrain, nTrain)
        """
        config = {"x_mean" : x_mean, "x_std" : x_std, "y_mean" : y_mean, "y_std" : y_std, 
                  "dydx_mean" : dydx_mean if cond else 0., "dydx_std" : dydx_std if cond else 1.}
        """

        config = {"x_mean" : torch.tensor(x_mean), "x_std" : torch.tensor(x_std), 
                  "y_mean" : torch.tensor(y_mean), "y_std" : torch.tensor(y_std), 
                  "dydx_mean" : torch.tensor(dydx_mean if cond else 0.), 
                  "dydx_std" : torch.tensor(dydx_std if cond else 1.) 
                  }

    else :
        config = {"x_mean" : 0.0, "x_std" : 1.0, "y_mean" : 0.0, "y_std" : 1.0, 
                  "dydx_mean" : 0.0, 
                  "dydx_std" : 1.0 }
    
    tensor_xTrain, tensor_yTrain = torch.FloatTensor(xTrain) , torch.FloatTensor(yTrain)
    tensor_xTest, tensor_yTest = torch.FloatTensor(xTest), torch.FloatTensor(yTest) 
    
    if with_derivative : 
        tensor_dydxTrain = torch.FloatTensor(dydxTrain)
        train_dataset = TensorDataset(tensor_xTrain, tensor_yTrain, tensor_dydxTrain)
        tensor_dydxTest = torch.FloatTensor(dydxTest)
        test_dataset = TensorDataset(tensor_xTest, tensor_yTest, tensor_dydxTest)
    else :
        train_dataset = TensorDataset(tensor_xTrain, tensor_yTrain)
        test_dataset = TensorDataset(tensor_xTest, tensor_yTest)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size) 
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size) 

    return train_dataloader, test_dataloader, xAxis, vegas, config
    
    