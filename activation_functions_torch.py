import torch
import torch.nn.functional as F

## softplus (formula : https://pytorch.org/docs/stable/nn.functional.html#softplus)
activation_function, deriv_activation_function = F.softplus, torch.sigmoid 

## Relu : ReLU(x) = max(0,x)
activation_function, deriv_activation_function = torch.relu, lambda x : (x > 0).float() 

## Gelu (formula : https://pytorch.org/docs/stable/nn.functional.html#gelu)
import math
a = math.sqrt(2/math.pi)
b = 0.044715
part = lambda x : 1 + torch.tanh(a*(x + b*(x**3)))
dpart = lambda x : (a*(1 + 3*b*(x)**2))*(1 - torch.tanh(a*(x + b*(x**3)))**2)
g = F.gelu # or lambda x : x * part(x) / 2
dg = lambda x : part(x) / 2 + x * dpart(x) / 2
activation_function, deriv_activation_function = g, dg

## Tanh
activation_function, deriv_activation_function = torch.tanh, lambda x : 1 - torch.tanh(x)**2 

## Sigmoid
g = torch.sigmoid
activation_function, deriv_activation_function = g, lambda x : g(x)*(1-g(x))

# Parameterised ReLU (formula : https://pytorch.org/docs/stable/nn.functional.html#prelu)
def prelu(a = 0.01):
    def f(x):
        # return torch.relu(x) + a * (x - abs(x))*0.5
        return F.prelu(input = x, weight = torch.tensor(a))
    return f
def prelu_deriv(a = 0.01):
    def f(x) :
        y = x.clone()
        for index in zip(*torch.nonzero(x > 0, as_tuple=True)):
            y[index] = 1.
        for index in zip(*torch.nonzero(0 >= x, as_tuple=True)):
            y[index] = a 
        return y
    return f

a = ?
activation_function, deriv_activation_function = prelu(a), prelu_deriv(a)

## Leaky ReLU : Parameterised ReLU with a = 0.01
activation_function, deriv_activation_function = prelu(a = 0.01), prelu_deriv(a = 0.01)

## Parameterised eLU (formula : https://pytorch.org/docs/stable/nn.functional.html#elu)
def elu(a = 1.):
    def f(x):
        return F.elu(input = x, alpha = torch.tensor(a))
    return f

def elu_deriv(a = 1.):
    def f(x) :
        y = x.clone()
        for index in zip(*torch.nonzero(x > 0, as_tuple=True)):
            y[index] = 1.
        for index in zip(*torch.nonzero(0 >= x, as_tuple=True)):
            y[index] = a * torch.exp(y[index])
        return y
    return f

a = ?
activation_function, deriv_activation_function = elu(a), elu_deriv(a)

## Elu : Parameterised eLU with a = 1
activation_function, deriv_activation_function = elu(a = 1.), elu_deriv(a = 1.)