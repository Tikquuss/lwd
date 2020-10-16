"""
try:
    %tensorflow_version 1.x
    %matplotlib inline
except Exception:
    pass
"""
print(tf.__version__)

# we want TF 1.x
assert tf.__version__ < "2.0"

import tensorflow as tf
import tensorflow.keras as K

## softplus
activation_function, deriv_activation_function = K.activations.softplus, K.activations.sigmoid

## Relu : ReLU(x) = max(0,x)
activation_function, deriv_activation_function = K.activations.relu, lambda x : tf.cast(x > 0, dtype = x.dtype)

## Gelu 
import math
a = math.sqrt(2/math.pi)
b = 0.044715
part = lambda x : 1 + K.activations.tanh(a*(x + b*(x**3)))
dpart = lambda x : (a*(1 + 3*b*(x)**2))*(1 - K.activations.tanh(a*(x + b*(x**3)))**2)
g = lambda x : x * part(x) / 2
dg = lambda x : part(x) / 2 + x * dpart(x) / 2
activation_function, deriv_activation_function = g, dg

## Tanh
activation_function, deriv_activation_function = K.activations.tanh, lambda x : 1 - K.activations.tanh(x)**2 

## Sigmoid
g = K.activations.sigmoid
activation_function, deriv_activation_function = g, lambda x : g(x)*(1-g(x))

# Parameterised ReLU 
def prelu(a = 0.01):
  @tf.function
  def f(x) :
      return tf.nn.relu(x) + a * (x - abs(x))*0.5
  return f

def prelu_deriv(a = 0.01):
    @tf.function
    def f(x) :
        """
        # generates graph errors
        y = tf.zeros_like(x)
        for index in zip(*tf.where(tf.greater(x, 0))): # x > 0
            y[index] = 1.
        for index in zip(*tf.where(tf.less_equal(x, 0))): # 0 >= x
            y[index] = a 
        return y
        """
        for_zero = tf.cast(tf.math.equal(x, 0), dtype = x.dtype)*a
        y = tf.cast(x > 0, dtype = x.dtype) +  a * (tf.cast(tf.math.not_equal(x, 0), dtype = x.dtype) - tf.math.sign(x)) * 0.5
        return y + for_zero

    return f

a = 0.01
activation_function, deriv_activation_function = prelu(a), prelu_deriv(a)

## Leaky ReLU : Parameterised ReLU with a = 0.01
activation_function, deriv_activation_function = prelu(a = 0.01), prelu_deriv(a = 0.01)

## Parameterised eLU 
def elu(a = 1.):
    @tf.function
    def f(x):
        #return K.activations.elu(x = x, alpha = torch.tensor(a))
        # This approximation is preferable as it also allows the derivative to be approximated.
        return tf.nn.relu(x) + a * (1 - tf.math.sign(x))*0.5*(tf.exp(x) - 1)
    return f

def elu_deriv(a = 1.):
    @tf.function
    def f(x) :
        """
        # generates graph errors
        y = tf.zeros_like(x)
        for index in zip(*tf.where(tf.greater(x, 0))): # x > 0
            y[index] = 1.
        for index in zip(*tf.where(tf.less_equal(x, 0))): # 0 >= x
            y[index] = a * tf.exp(x[index])
        return y
        """
        for_zero = tf.cast(tf.math.equal(x, 0), dtype = x.dtype)*a
        y = tf.cast(x > 0, dtype = x.dtype) + a * (tf.cast(tf.math.not_equal(x, 0), dtype = x.dtype) - tf.math.sign(x))*0.5*tf.exp(x)
        return y + for_zero
    return f

a = 1.
activation_function, deriv_activation_function = elu(a), elu_deriv(a)

## Elu : Parameterised eLU with a = 1
activation_function, deriv_activation_function = elu(a = 1.), elu_deriv(a = 1.)