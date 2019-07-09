import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from autograd import grad

#Generating Data
np.random.seed(0)
n_data = 50
X = np.linspace(1, 50, n_data)
Y = 7*X + 9 + 2*np.random.randn(n_data)


#Model to be learnt y=7x+9 from y=Wx+b;

def cost(param_list):
    w, b = param_list
    pred = w*X+b
    return np.sqrt(((pred - Y) ** 2).mean(axis=None))/(2*len(Y)) 

def adagrad_gd(param_init, cost, n, lr, eps):
    from copy import deepcopy
    grad_cost = grad(cost)
    params = deepcopy(param_init)
    param_array, grad_array, lr_array, cost_array = [params], [], [[lr for _ in params]], [cost(params)]
    sum_squares_gradients = [np.zeros_like(param) for param in params]
    for i in range(n):
        out_params = []
        gradients = grad_cost(params)
        	
	# At each iteration, we add the square of the gradients to sum_squares_gradients
        sum_squares_gradients= [eps + sum_prev + np.square(g) for sum_prev, g in zip(sum_squares_gradients, gradients)]
        
	# Adapted learning rate for parameter list
        lrs = [np.divide(lr, np.sqrt(sg)) for sg in sum_squares_gradients]
        
	# Paramter update
        params = [param-(adapted_lr*grad_param) for param, adapted_lr, grad_param in zip(params, lrs, gradients)]
        param_array.append(params)
        	
	lr_array.append(lrs)
        grad_array.append(gradients)
        cost_array.append(cost(params))
        
    return params, param_array, grad_array, lr_array, cost_array

np.random.seed(0)
param_init = [np.random.randn(), np.random.randn()]
lr = 0.1
eps=1e-8
n=1000
ada_params, ada_param_array, ada_grad_array, ada_lr_array, ada_cost_array = adagrad_gd(param_init, cost, n, lr, eps)
