import math
import numpy as np
import random

import jax
import jax.numpy as jnp

import torch

# 1) Styblinski-Tang function : https://www.sfu.ca/~ssurjano/stybtang.html 

def STFunction(x, d=2):
    val = 0
    for i in range(d):
        val += x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i]
    val *= 0.5
    return val

def f(x) :
    if len(x.shape) == 1 : 
        return 0.5 * jnp.sum(a = x**4 - 16*x**2 + 5*x)
        #return 0.5 * torch.sum(input = x**4 - 16*x**2+5*x)
    else : # batch
        #return 0.5 * torch.sum(input = x**4 - 16*x**2+5*x, dim = 1)
        return 0.5 * jnp.sum(a = x**4 - 16*x**2 + 5*x, axis = 1)
    
def STDeriv(index):
    def f(x):
        val = 0.5 * (4 * x[index] ** 3 - 32 * x[index] + 5)
        return val
    return f
    
# 2) Ackley function : http://www.sfu.ca/~ssurjano/ackley.html 

def AckleyFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
    part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
    return  math.exp(1) + 20 -20*math.exp(part_1) - math.exp(part_2)

def AckleyDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]

        part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
        part_1 = -20*math.exp(part_1)
        coef_deriv_part_1 = -0.2*math.sqrt(0.5)*x[index] / math.sqrt(x1*x1 + x2*x2)

        part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
        part_2 =  - math.exp(part_2)
        coef_deriv_part_2 = 0.5*(-2*math.pi*math.sin(2*math.pi*x[index]))

        return coef_deriv_part_1*part_1 + coef_deriv_part_2*part_2

    return f
    
# 3) Beale function : https://www.sfu.ca/~ssurjano/beale.html
def BealeFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = (1.5 - x1 + x1*x2)**2
    part_2 = (2.25 - x1 + x1*(x2**2))**2
    part_3 = (2.625 - x1 + x1*(x2**3))**2
    return  part_1 + part_2 + part_3

def BealeDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        
        part_1 = 1.5 - x1 + x1*x2
        part_2 = 2.25 - x1 + x1*(x2**2)
        part_3 = 2.625 - x1 + x1*(x2**3)

        if index == 0 :
            return 2*(-1 + x2)*part_1 + 2*(-1 + x2**2)*part_2 + 2*(-1 + x2**3)*part_3
        elif index == 1 :
            return 2*x1*part_1 + 2*(2*x1*x2)*part_2 + 2*(3*x1*(x2**2))*part_3 

    return f
    
# 4) Booth function : https://www.sfu.ca/~ssurjano/booth.html
def BoothFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = (x1 + 2*x2 - 7)**2
    part_2 = (2*x1 + x2 - 5)**2
    return  part_1 + part_2
    
def BoothDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        
        part_1 = (x1 + 2*x2 - 7)
        part_2 = (2*x1 + x2 - 5)
      
        if index == 0 :
            return 2*part_1 + 4*part_2 
        elif index == 1 :
            return 4*part_1 + 2*part_2 

    return f
    
# 5) Bukin function : https://www.sfu.ca/~ssurjano/bukin6.html

def part_1(x):
    x1, x2 = x[0], x[1] 
    return math.sqrt(abs(x2 - 0.01*(x1**2)))

def part_2(x):
    x1, x2 = x[0], x[1]
    return abs(x1 + 10)

def BukinFunction(x):
    return 100*part_1(x) + 0.01*part_2(x)

def part_1_deriv(x, index):
    x1, x2 = x[0], x[1]
    try : 
        if index == 0 :
            condition = x2 > 0 and -math.sqrt(x2/0.01) < x1 < math.sqrt(x2/0.01)
            return (-1 if condition else 1)*0.01*x1/part_1(x)
        elif index == 1 :
            return (-1 if x2 < 0.01*(x1**2) else 1)/(2*part_1(x)) 

    except ZeroDivisionError :
        assert x2 == 0.01*(x1**2)
        return 0

def part_2_deriv(x, index):
    if index == 0 :
        return 1 if x[0] > -10 else -1 
    elif index == 1 :
        return 0

def BukinDeriv(index):
    def f(x):
        return 100*part_1_deriv(x, index) + 0.01*part_2_deriv(x, index)
    return f
    
# 6) McCormick function : https://www.sfu.ca/~ssurjano/mccorm.html

def McCormickFunction(x):
    x1, x2 = x[0], x[1]
    return  math.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1
    
def McCormickDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        if index == 0 :
            return math.cos(x1 + x2) + 2*(x1 - x2) - 1.5
        elif index == 1 :
            return math.cos(x1 + x2) - 2*(x1 - x2) + 2.5

    return f
    
# 7) Rosenbrock function : https://www.sfu.ca/~ssurjano/mccorm.html

def RosenbrockFunction(x):
    x1, x2 = x[0], x[1]
    return 100*(x2-x1**2)**2 + (x1 - 1)**2 
    
def RosenbrockDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        if index == 0 :
            return -400*x1*(x2 - x1**2) + 2*(x1 - 1)
        elif index == 1 :
            return 200*(x2 - x1**2)

    return f
