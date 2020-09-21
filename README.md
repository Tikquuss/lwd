This repository contains the code for :
* [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)
* [Differential Machine Learning](https://arxiv.org/abs/2005.02347) 
* [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

Test on different functions :
* Those used in Sobolev Training paper (see this [notebook](notebooks/optimization_functions.ipynb)) :
    * [Styblinski-Tang](https://www.sfu.ca/~ssurjano/stybtang.html)
    * [Ackley](http://www.sfu.ca/~ssurjano/ackley.html)
    * [Beale](https://www.sfu.ca/~ssurjano/beale.html)
    * [Booth](https://www.sfu.ca/~ssurjano/booth.html)
    * [Bukin](https://www.sfu.ca/~ssurjano/bukin6.html)
    * [McCormick](https://www.sfu.ca/~ssurjano/mccorm.html)
    * [Rosenbrock](https://www.sfu.ca/~ssurjano/rosen.html)

* Those used in Differential Machine Learning paper (see this [notebook](notebooks/financial_functions.ipynb)) :
    * Pricing and Risk Functions : Black & Scholes ([wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model))
    * Gaussian Basket options : Bachelier dimension 1, 7, 20...([mathworks](https://www.mathworks.com/help/finance/pricing-american-basket-options-by-monte-carlo-simulation.html))

To summarize, you have the possibility to form a network following one of these processes :
* Normal Training (x, y) : with MLP (multi-layer perceptron) and Siren
* Sobolev Training (x, y, dy/dx) : with MLP and Siren
* Twin_net tensorflow (x, y, dy/dx) : with MLP and Siren
* Twin_net pytorch (x, y, dy/dx) : with MLP and Siren

# **How to train your one model?**
```
git clone https://github.com/Tikquuss/lwd
cd lwd/scripts
```
## **(i) For achitecture made with pytorch (Normal Training, Sobolev Training and Twin_net pytorch): [utils.py](scripts/utils.py) and [functions.py](scripts\functions.py)**

### **Data**  

The genData function takes a :
* function : f(x : array), return y 
* its derivative: f'(i: int), takes i as parameter and returns another function that takes x and returns df(x)/dx[i]=dy/dx[i].
* dim_x : dimension of x
* the boundaries of the domain in which the points will be generated unify the points
* and the number of examples (n) to be generated

and returns (xi, yi, [dydx[j], j=1...dim_x]), i = 1….n

**Example with the Styblinski-Tang function**
```
from utils import genData, get_data_loader
from functions import STFunction, STDeriv

min_x, max_x = -5, 5
batch_size = 32
normalize = False # whether you want to normalize the data or not.
nTrain = number of examples to be generated for training
nTest = number of examples to be generated for the test
INPUT_DIM = 2

batch_samples = genData(function = STFunction, deriv_function = STDeriv, dim_x = INPUT_DIM, min_x = min_x, max_x = max_x, num_samples = nTrain)
x = [bs[0] for bs in batch_samples]
y = [bs[1] for bs in batch_samples]
dydx = [bs[2] for bs in batch_samples]
train_dataloader, config = get_data_loader(x = x, y = y, dydx = dydx_train, batch_size = batch_size, normalize = normalize)

batch_samples = genData(function = STFunction, deriv_function = STDeriv, dim_x = INPUT_DIM, min_x = min_x, max_x = max_x, num_samples = nTest)
x = [bs[0] for bs in batch_samples]
y = [bs[1] for bs in batch_samples]
dydx = [bs[2] for bs in batch_samples]
test_dataloader, _ = get_data_loader(x = x, y = y, dydx = dydx_train, batch_size = batch_size, normalize = False)
```

* **Note: case of financial functions (Black & Scholes, Bachelier)**
```
from twin_net_tf import get_diffML_data_loader, BlackScholes, Bachelier

generator = BlackScholes() # or Bachelier(n = INPUT_DIM) for Bachelier dimension INPUT_DIM
train_seed, test_seed = 0, 0 

train_dataloader, test_dataloader, _, _, config = get_diffML_data_loader(
        generator = generator, 
        nTrain = nTrain, nTest = nTest, 
        train_seed = train_seed, test_seed = test_seed, 
        batch_size = batch_size, with_derivative = with_derivative,
        normalize = normalize
    )
```

If `normalize = True`, config will be a dictionary containing the following key-value pairs:
- "x_mean": mean of x
- "x_std" : variance of x
- "y_mean": mean of y
- "y_std" : variance of y
- "dydx_mean": mean of dydx
- "dydx_std" : variance of dydx

If you are in dimension 2, you can visualize the curve of your function and its deviation as follows:
```
from utils import plotFunction, plotGrad

min_y, max_y =  -5, 5 
step_x, step_y = 0.25, 0.25

plotFunction(name = "Styblinski-Tang Function", function = STFunction, 
             min_x = min_x, max_x = max_x, step_x = step_x, 
             min_y = min_y, max_y = max_y, step_y = step_y)

plotGrad(name = "Gradient Field of Styblinski-Tang Function", deriv_function = STDeriv, 
         min_x = min_x, max_x = max_x, step_x = step_x, 
         min_y = min_y, max_y = max_y, step_y = step_y)
```

* **hyperparameters in the different loss functions to express a tradeoff between y loss and dydx loss**
``` 
loss_config = {'alpha': 1, "beta" : 1} # loss = alpha * loss_y + beta * loss_dydx
config.update(loss_config)
```

Savine et al. applied the recent [one-cycle learning rate schedule](https://arxiv.org/abs/1803.09820) of Leslie Smith and found that it considerably accelerates and stabilizes the training of neural networks.
This parameter was introduced for this purpose, and remains optional (so you can override these two lines of code, and the learning rate will be used as described below).
```
learning_rate_schedule = [(0.0, 1.0e-8), (0.2, 0.1), (0.6, 0.01), (0.9, 1.0e-6), (1.0, 1.0e-8)]
config["learning_rate_schedule"] = learning_rate_schedule
```
### **Model**

* *Parameters of the model*

```
INPUT_DIM = 2 
HIDDEN_DIM = 20
N_HIDDEN = 2 # number of hidden layers
OUTPUT_DIM = 1
```

* *case of multi-layer perceptron*

```
import torch
import torch.nn.functional as F
from utils MLP

activation_function = F.softplus
deriv_activation_function = torch.sigmoid # for twin_net (backprop)
mlp_model_kwargs = {"in_features" : INPUT_DIM, # depends on the function
                    "hidden_features" : HIDDEN_DIM, 
                    "hidden_layers" : N_HIDDEN, 
                    "out_features": OUTPUT_DIM, 
                    "activation_function" : activation_function, 
                    "deriv_activation_function" : deriv_activation_function,
                   }
model = MLP(**mlp_model_kwargs)
```
* *case of Siren*
```
from utils import Siren
first_omega_0 = 30.
hidden_omega_0 = 30.
outermost_linear = True

siren_model_kwargs = {"in_features" : INPUT_DIM, 
                      "hidden_features" : HIDDEN_DIM, 
                      "hidden_layers" : N_HIDDEN, 
                      "out_features": OUTPUT_DIM, 
                      "outermost_linear" : outermost_linear, 
                      "first_omega_0" : first_omega_0, 
                      "hidden_omega_0" : hidden_omega_0}

model = Siren(**siren_model_kwargs)
```

### **Optimizer and loss function**
```
learning_rate = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.MSELoss()
```
### **Training, visualization of training statistics and testing**
```
from utils import train, plot_stat, test

name = "net" for normal and sobolev training, "twin_net" for twin_net (do not specify any other value than the last two)

with_derivative = False for normal training, True for sobolev training and twin_net 

max_epoch = # maximun number of epoch
improving_limit = float("inf") # Stop training if the training loss does not decrease n times (no limit here)

model, stats, best_loss = train(
    name, model, train_dataloader, 
	optimizer, 
	criterion, 
	config, 
    with_derivative, max_epoch = max_epoch, 
    improving_limit = improving_limit
)


plot_stat(stats, with_derivative = with_derivative)

(test_loss, r_y, r_dydx), (x_list, y_list, dydx_list, y_pred_list, dydx_pred_list) = test(
        name, model, test_dataloader, criterion, config, with_derivative
    )
```
If you are in dimension 2 and want to visualize the curves produced by your models :

```
from utils import forward, backprop

x_mean, x_std = config.get("x_mean", 0.), config.get("x_std", 1.)
y_mean, y_std = config.get("y_mean", 0.), config.get("y_std", 1.)
dydx_mean, dydx_std = config.get("dydx_mean", 0.), config.get("dydx_std", 1.)

def function(x):
    x = torch.tensor(x)
    x_scaled = (x-x_mean) / x_std
    y_pred_scaled = model(x = x_scaled.float())
    y_pred = y_mean + y_std * y_pred_scaled
    y_pred = y_pred.detach().squeeze().numpy()
    return y_pred

def deriv_function(index):
    def f(x) :
        x = torch.tensor(x, requires_grad = True)
        x_scaled = (x-x_mean) / x_std
        if name == "net" :
            y_pred_scaled = model(x = x_scaled.float()) 
            dydx_pred_scaled = gradient(y_pred_scaled, x_scaled)
        elif name == "twin_net" :
            y_pred_scaled, zs = forward(net = model.net, x = x_scaled.float(), return_layers = True)
            dydx_pred_scaled = backprop(net = model.net, y = y_pred_scaled, zs = zs)
        dydx_pred = dydx_mean + dydx_std * dydx_pred_scaled
        dydx_pred = dydx_pred.detach().squeeze().numpy()
        return dydx_pred[index]
    return f


plotFunction(name = "Styblinski-Tang Function foo foo", function = function, 
             min_x = min_x, max_x = max_x, step_x = step_x, 
             min_y = min_y, max_y = max_y, step_y = step_y)

plotGrad(name = "Gradient Field of Styblinski-Tang Function foo foo", deriv_function = deriv_function, 
             min_x = min_x, max_x = max_x, step_x = step_x, 
             min_y = min_y, max_y = max_y, step_y = step_y)
```
* **Note: case of financial functions**
```
import matplotlib.pyplot as plt

xy = [(x[0], y[0]) for x, y in zip(x_list, y_list)]
xy_pred = [(x[0], y[0]) for x, y in zip(x_list, y_pred_list)]
  
if with_derivative :
    xdydx = [(x[0], y[0]) for x, y in zip(x_list, dydx_list)]
    xdydx_pred = [(x[0], y) for x, y in zip(x_list, dydx_pred_list)]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize = (15,3))
else :
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize = (15,3))

fig.suptitle('foo')
    
ax1.scatter(*zip(*xy), label = "y")
ax1.scatter(*zip(*xy_pred), label = "ypred")
ax1.set(xlabel='x', ylabel='y, y_pred')
ax1.legend()
if with_derivative :
    ax2.scatter(*zip(*xdydx), label = "dy")
    ax2.scatter(*zip(*xdydx_pred), label = "dy pred")
    ax2.set(xlabel='x', ylabel='dy, dy_pred')
    ax2.legend()
```

## **(ii) For achitecture made with tensorflow (Twin_net tensorflow) : [utils.py](), [twin_net_tf.py]() and [twin_net_tf_siren.py]()**

### **Generator to be used to produce training and test data.**

* **In general** :
```
from twin_net_tf import Generator
from functions import STFunction, STDeriv

min_x, max_x = -5, 5
INPUT_DIM = 2
generator = Generator(callable_function = STFunction, 
                      callable_function_deriv = STDeriv, 
                      dim_x = INPUT_DIM,
                      min_x = min_x, max_x = max_x)
```
* **For financial functions**
```
from twin_net_tf import BlackScholes, Bachelier
generator = BlackScholes() # or Bachelier(n = INPUT_DIM) for Bachelier dimension INPUT_DIM
```

### **Training, testing and visualization of training statistics**
```
from twin_net_tf import graph, test as twin_net_tf_test 
from twin_net_tf_siren import test as twin_net_tf_test_siren

siren = False # set to True if you want to use siren as backbone
nTrain = number of examples to be generated for training
nTest = number of examples to be generated for the test
train_seed, test_seed = 0, 0

HIDDEN_DIM = 20
N_HIDDEN = 2
generator_kwargs = {"hidden_units" : HIDDEN_DIM, 
                    "hidden_layers" : N_HIDDEN}

max_epoch = # maximun number of epoch
improving_limit = float("inf") # Stop training if the training loss does not decrease n times (no limit here)

if siren :
    first_omega_0 = 30.
    hidden_omega_0 = 30.
    outermost_linear = True
    loss, regressor, dtrain, dtest, dydxTest, values, deltas = twin_net_tf_test_siren(
              generator, [nTrain], 
              nTrain, nTest, 
              trainSeed = train_seed, testSeed = test_seed, weightSeed = 0, 
              deltidx=0,
              generator_kwargs = generator_kwargs,
              epochs = max_epoch,
              first_omega_0 = first_omega_0, 
              hidden_omega_0 = hidden_omega_0, 
              outermost_linear = outermost_linear,
              improving_limit = improving_limit
          )
else :
    loss, regressor, dtrain, dtest, dydxTest, values, deltas = twin_net_tf_test(
              generator, [nTrain], 
              nTrain, nTest, 
              trainSeed = train_seed, testSeed = test_seed, weightSeed = 0, 
              deltidx = 0,
              generator_kwargs = generator_kwargs,
              epochs = max_epoch,
              improving_limit = improving_limit
          )

#plot_stat(regressor.stats["normal"], with_derivative = with_derivative)
#plot_stat(regressor.stats["differential"], with_derivative = with_derivative)
```
If you are in dimension 2 and want to visualize the curves produced by your models :
```
plotFunction(name = "Styblinski-Tang Function foo foo", function =  lambda x : regressor.predict_values([x])[0][0], 
                 min_x = min_x, max_x = max_x, step_x = step_x, 
                 min_y = min_y, max_y = max_y, step_y = step_y)

plotGrad(name = "Gradient Field of Styblinski-Tang Function foo foo", 
            deriv_function = lambda index : lambda x : regressor.predict_values_and_derivs([x])[1][0][index], 
            min_x = min_x, max_x = max_x, step_x = step_x, 
            min_y = min_y, max_y = max_y, step_y = step_y)


# show_graph_per_axis
yTest = dtest[1]
for i in range(INPUT_DIM) :
    xAxis  = np.array([[x[i]] for x in dtest[0]])
    # show predicitions
    graph("%s x%d vs y" % (name, (i+1)), values, xAxis, "", "values", yTest, [nTrain], True)
    # show deltas
    graph("%s x%d vs dxdy" % (name, (i+1)), deltas, xAxis, "", "deltas", dydxTest, [nTrain], True)
```
For financial functions
```
yTest = dtest[1]
sizes = [nTrain]
# show predicitions
graph(name, values, xAxis, "", "values", yTest, [nTrain], True)
# show deltas
graph(name, deltas, xAxis, "", "deltas", dydxTest, [nTrain], True)

# show_graph_per_axis 
for i in range(INPUT_DIM) :
    xAxis = np.array([[x[i]] for x in dtest[0]])
    # show predicitions
    graph("%s x%d vs y" % (name, (i+1)), values, xAxis, "", "values", yTest, [nTrain], True)
    # show deltas
    graph("%s x%d vs dxdy" % (name, (i+1)), deltas, xAxis, "", "deltas", dydxTest, [nTrain], True)
```

# References

* https://github.com/mcneela/Sobolev
* https://github.com/differential-machine-learning/notebooks
* https://github.com/vsitzmann/siren