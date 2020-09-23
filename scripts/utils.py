import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from tqdm import tqdm_notebook
import random
import os 
import itertools

#from twin_net_tf import normalize_data

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    """    
    else :
        a = grad_outputs.shape == y[0].shape
        assert a or grad_outputs.shape == y.shape
        if a :
             grad_outputs = grad_outputs.repeat(y.shape[0], 1, 1)
    """
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def genData(function, deriv_function, dim_x, min_x, max_x, num_samples):
    samples = []
    for n in range(num_samples):
        x = np.array([random.uniform(min_x, max_x) for i in range(dim_x)])
        y = function(x)
        dy = np.array([deriv_function(i)(x) for i in range(dim_x)])
        s = (x, y, dy)
        samples.append(s)
    return samples

def plotFunction(name, function = None, model = None, 
                 min_x = -5, max_x = 5, step_x = 0.25, 
                 min_y = -5, max_y = 5, step_y = 0.25) :
    assert function or model
    x = np.arange(start = min_x, stop = max_x, step = step_x, dtype = np.float)
    y = np.arange(start = min_y, stop = max_y, step = step_y, dtype = np.float)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])

    X = np.array(X)
    z = []

    if model :
        model.eval()
        X = torch.FloatTensor(X)
        ds = TensorDataset(X, torch.ones_like(X)) 
        dsloader = DataLoader(ds, batch_size = 1)
        with torch.no_grad():
            for batch, _ in dsloader :
                z.append(model(batch)[0].squeeze().numpy())
    else :    
        for k in range(len(X)):
            z.append(function(X[k]))


    z = np.array(z).reshape((len(x), len(x[0])))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(name)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plotGrad(name, deriv_function = None, model = None, 
             min_x = -5, max_x = 5, step_x = 0.25, 
             min_y = -5, max_y = 5, step_y = 0.25):
    assert deriv_function or model
    x = np.arange(start = min_x, stop = max_x, step = step_x, dtype = np.float)
    y = np.arange(start = min_y, stop = max_y, step = step_y, dtype = np.float)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])
    X = np.array(X)
    z = []

    if model :
        model.train()
        X = torch.FloatTensor(X)
        dsloader = DataLoader(TensorDataset(X, torch.ones_like(X)), batch_size = 1)
        for batch, _ in dsloader :
            z.append(gradient(*model(batch))[0].detach().numpy())
    else :
        grad1 = deriv_function(index = 0)
        grad2 = deriv_function(index = 1)

        for k in range(len(X)):
            z.append([grad1(X[k]), grad2(X[k])])
    
    z = np.array(z)
    z = np.array(z).reshape((len(x), len(x[0]), 2))

    fig = plt.figure()
    plt.title(name)
    dz = plt.quiver(x, y, z[:, :, 0], z[:, :, 1])
    plt.show()

def plot_stat(stats, with_derivative = False):
    if with_derivative :
        fig, (ax2, ax3) = plt.subplots(1, 2, sharex=True, figsize = (20, 3))
        fig.suptitle('')

        ax2.plot(range(len(stats['train_yloss'])), stats['train_yloss'], label='train')
        ax2.set(xlabel='epoch', ylabel='yloss')
        ax2.set_title('yloss per epoch')
        ax2.legend()
        #ax2.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax3.plot(range(len(stats['train_dyloss'])), stats['train_dyloss'], label='train')
        ax3.set(xlabel='epoch', ylabel='dyloss')
        ax3.set_title('dyloss per epoch')
        ax3.legend()
        #ax3.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

    else :
        plt.plot(range(len(stats['train_loss'])), stats['train_loss'])

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
        dydx_mean = dycropped_dxcropped.mean(axis=0)
        dydx_std = dycropped_dxcropped.std(axis=0) + epsilon
        dydx = (dycropped_dxcropped - dydx_mean) / dydx_std
    else:
        dydx_mean = None
        dydx_std = None
        dydx = None
    
    return (x_mean, x_std, x), (y_mean, y_std, y), (dydx_mean, dydx_std, dydx)

def get_data_loader(x, y, dydx = None, batch_size = 32, normalize = False):

    if not normalize :
  
        tensor_x = torch.FloatTensor(x)  
        tensor_y = torch.FloatTensor(y)

        try : 
            if dydx: 
                tensor_dydx = torch.FloatTensor(dydx)
                dataset = TensorDataset(tensor_x, tensor_y, tensor_dydx)
            else :
                dataset = TensorDataset(tensor_x, tensor_y)
        except ValueError :
            if dydx.all(): 
                tensor_dydx = torch.FloatTensor(dydx)
                dataset = TensorDataset(tensor_x, tensor_y, tensor_dydx)
            else :
                dataset = TensorDataset(tensor_x, tensor_y)

        dataloader = DataLoader(dataset, batch_size = batch_size) 

        _, n = np.array(x).shape
        config = {"x_mean" : 0.0, "x_std" : 1.0, "y_mean" : 0.0, "y_std" : 1.0, "lambda_j" : 1.0, "n" : n}

        return dataloader, {}
        
    else :
        m = len(x)
        x = np.array(x)
        y = np.array(y)
        cond = not (dydx is None)
        dydx = np.array(dydx) if cond else None

        _, n = x.shape
        
        (x_mean, x_std, x), (y_mean, y_std, y), (dydx_mean, dydx_std, dydx) = normalize_data(x, y, dydx, None)
        """
        config = {"x_mean" : x_mean, "x_std" : x_std, "y_mean" : y_mean, "y_std" : y_std, 
                  "dydx_mean" : dydx_mean if cond else 0., "dydx_std" : dydx_std if cond else 1.}
        """

        config = {"x_mean" : torch.tensor(x_mean), "x_std" : torch.tensor(x_std), 
                  "y_mean" : torch.tensor(y_mean), "y_std" : torch.tensor(y_std), 
                  "dydx_mean" : torch.tensor(dydx_mean if cond else 0.), 
                  "dydx_std" : torch.tensor(dydx_std if cond else 1.) 
                  }
        

        dataloader, _ = get_data_loader(x, y, dydx, batch_size, normalize = False)

        return dataloader, config
            

def forward(net, x, return_layers = False):
    if len(x.shape) == 1 : 
        x = x.reshape(1, x.shape[0]) # batching

    if not return_layers :
        for linear_layer in net :
            # x = g_l ( zl-1 @ (omega_l * wl.T) + bl )
            x = linear_layer.activation_function(x @ (linear_layer.omega_0 * linear_layer.linear.weight.t()) + linear_layer.linear.bias) 
        return x
    else :
        zs = []
        for linear_layer in net :
            # zl_stilde = zl-1 @ (omega_l * wl.T) + bl 
            z = x @ (linear_layer.omega_0 * linear_layer.linear.weight.t()) + linear_layer.linear.bias 
            zs.append(z)
            # x = g_l ( zl_stilde )
            x = linear_layer.activation_function(z) 
        return x, zs

def backprop(net, y, zs, vL = None):
    ########### 1 #############
    m, n = y.shape[-1], y.shape[0]

    if vL is None :
        vL = torch.ones_like(y) # [[1, ...1], [1, 1, ...1] ... [1, ...1]]
    else :
        a = vL.shape == y[0].shape # [v1, ...]
        assert a or vL.shape == y.shape # a or [[v11, ...], [v21, , ...] ...]
        if a :
            vL = vL.repeat(n, 1, 1) # [[v1, ...], [v1, , ...] ...]
    
    ybar = torch.eye(m)
    if len(y.shape) != 1 :
        ybar = ybar.repeat(n, 1, 1) # [[1, ...1], [1, 1, ...1] ... [1, ...1]]

    zbar = torch.bmm(input = vL.reshape(n, 1, m), mat2 = ybar) # = vL if vL = [[1, ...1], [1, 1, ...1] ... [1, ...1]]

    L = len(zs)
    for l in range(L-1, -1, -1):
        linear_layer = net[l] 
        jacobian_of_dgdzl_stilde = torch.stack(
            [torch.diag(linear_layer.deriv_activation_function(zs[l][i])) for i in range(n)  # zL-1, ...,z0
        ]) 
        zbar = torch.bmm(input = torch.bmm(
                                        input = zbar, 
                                        mat2 = jacobian_of_dgdzl_stilde
                                  ), 
                          mat2 = torch.stack([linear_layer.omega_0 * linear_layer.linear.weight for i in range(n)])
                          ) # zl-1_bar = zl_bar @ g'(zl_stilde) @ wl
        zbar.detach_()
    xbar = zbar.squeeze() # z0_bar
    return xbar
    
    """    
    L = len(zs)
    xbars = []
    for i, y in enumerate(ys) :
        vL = torch.ones_like(y)
        zL = torch.eye(y.shape[0])
        zbar = vL @ zL
        for l in range(L-1, -1, -1):
            linear_layer = self.net[l] 
            jacobian_of_dgdzl_stilde = torch.diag(linear_layer.deriv_activation_function(zs[l][i]))
            z = linear_layer.omega_0 * linear_layer.linear.weight
            zbar = zbar @ jacobian_of_dgdzl_stilde @ z
            zbar.detach_()    
        xbars.append(zbar)

    return torch.stack(xbars)
    """
    
    ############## 2 #############
    """
    if not vL :
        vL = torch.ones_like(y) 
    else :
        assert vL.shape == y.shape
   
    xbars = [] 
    for i, (y_tmp, vL_tmp) in enumerate(zip(y, vL)) :
        ybar = torch.eye(y_tmp.shape[0]) # [[1, ...0], [0, 1, ...0] ... [0, ...1]]
        zbar = vL_tmp @ ybar # [1, ...., 1]
        L = len(zs)
        for l in range(L-1, -1, -1):
            linear_layer = net[l] 
            jacobian_of_dgdzl_stilde = torch.diag(linear_layer.deriv_activation_function(zs[l][i])) # g'(zl_stilde)
            z = linear_layer.omega_0 * linear_layer.linear.weight
            zbar = zbar @ jacobian_of_dgdzl_stilde @ z # zl-1_bar = zl_bar @ g'(zl_stilde) @ wl
            zbar.detach_()
        xbars.append(zbar)
    xbars = torch.stack(xbars)
    return xbars
    """
    ############# 3 ###########
    """
    if not vL :
        vL = torch.ones_like(y) # [1, ...., 1]
    else :
        assert vL.shape == y.shape

    ybar = torch.eye(y.shape[0]) # [[1, ...0], [0, 1, ...0] ... [0, ...1]]
    zbar = vL @ ybar # [1, ...., 1]

    L = len(zs)
    for l in range(L-1, -1, -1):
        linear_layer = net[l] 
        jacobian_of_dgdzl_stilde = torch.diag(linear_layer.deriv_activation_function(zs[l])) # g'(zl_stilde)
        zbar = zbar @ jacobian_of_dgdzl_stilde @ linear_layer.linear.weight # zl-1_bar = zl_bar @ g'(zl_stilde) @ wl
        zbar.detach_()
    xbar = zbar # z0
    return xbar
    """

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias = True, activation_function = None, deriv_activation_function = None, omega_0 = 1.):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.activation_function = activation_function if activation_function else lambda x : x
        self.deriv_activation_function = deriv_activation_function if deriv_activation_function else lambda x : x
        self.omega_0 = omega_0

    def forward(self, x):
        return self.activation_function(self.omega_0 * self.linear(x))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                        activation_function = None, deriv_activation_function = None,
                        first_omega_0 = 1., hidden_omega_0 = 1.):
        super(MLP, self).__init__()
        net = []
        net.append(Linear(in_features, hidden_features, True, activation_function, deriv_activation_function, first_omega_0))  
        net += [Linear(hidden_features, hidden_features, True, activation_function, deriv_activation_function, hidden_omega_0) for _ in range(hidden_layers)]  
        net.append(Linear(hidden_features, out_features, True, None, None, hidden_omega_0))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
   
class Siren(MLP):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear = False, 
                 first_omega_0 = 30., hidden_omega_0 = 30.):

        super(Siren, self).__init__(in_features, hidden_features, hidden_layers, out_features, 
                                    torch.sin, torch.cos, first_omega_0 = 30., hidden_omega_0 = 30.)
        if outermost_linear :
            self.net[-1] = Linear(hidden_features, out_features, True, torch.sin, torch.cos)

        # init_weights
        with torch.no_grad():
            self.net[0].linear.weight.uniform_(-1 / in_features, 1 / in_features)      
            
            for l in range(1, len(self.net)) :
                self.net[l].linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                                    np.sqrt(6 / hidden_features) / hidden_omega_0)

def train(name, model, dataloader, optimizer, criterion, config, with_derivative, max_epoch, improving_limit = float('inf'), metric = "train_loss"):
    
    assert name in ["net", "twin_net"]

    assert all([model, dataloader, criterion])

    description = config.get("description", "train...")
    y_mean, y_std = config.get("y_mean", 0.), config.get("y_std", 1.)
    dydx_mean, dydx_std = config.get("dydx_mean", 0.), config.get("dydx_std", 1.)
    alpha, beta = config.get("alpha", 1.), config.get("beta", 1.)
    learning_rate_schedule = config.get("learning_rate_schedule", None)
    if learning_rate_schedule :
        lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)

    if with_derivative :
        assert metric in ["train_loss", "train_yloss", "train_dyloss"]
    else :
        assert metric == "train_loss"

    len_dl = len(dataloader)
    assert len_dl

    model.train()
    
    tmp_best_model_path = './best-model-tmp.pth'
    best_loss = float('inf')
    counter = 1
    stats = {}
    
    if with_derivative :
        stats['train_loss'] = []
        stats['train_yloss'] = []
        stats['train_dyloss'] = []

        for epoch in tqdm_notebook(range(max_epoch), desc=description):
            if learning_rate_schedule :
                # interpolate learning rate in cycle
                learning_rate = np.interp(epoch / max_epoch, lr_schedule_epochs, lr_schedule_rates)
                optimizer.lr = learning_rate

            running_loss = 0
            r_y, r_dydx = 0, 0

            r_y_no_scaled, r_dydx_no_scaled = 0, 0
              
            for batch in dataloader:
              
                x, y, dydx = batch

                optimizer.zero_grad()
                
                if name == "net" :
                    # Forward pass
                    x.requires_grad_(True)
                    y_pred = forward(net = model.net, x = x)
                    # Compute gradient 
                    dydx_pred = gradient(y_pred, x)
                else :
                    # Forward pass
                    y_pred, zs = forward(net = model.net, x = x, return_layers = True)
                    # Compute gradient 
                    dydx_pred = backprop(net = model.net, y = y_pred, zs = zs)

                # Compute Loss
                l_y = criterion(y_pred.squeeze(), y)
                l_dydx = criterion(dydx, dydx_pred.detach())

                loss = alpha * l_y + beta * l_dydx
                
                running_loss += loss.item()
                r_y += l_y.item()
                r_dydx += l_dydx.item()

                y_pred = y_mean + y_std * y_pred
                dydx_pred = dydx_mean + dydx_std * dydx_pred

                y = y_mean + y_std * y
                dydx = dydx_mean + dydx_std * dydx

                l_y_no_scaled = criterion(y_pred.squeeze(), y).item()
                l_dydx_no_scaled = criterion(dydx , dydx_pred).item()

                r_y_no_scaled += l_y_no_scaled
                r_dydx_no_scaled += l_dydx_no_scaled
                
                # Backward pass
                loss.backward()
                optimizer.step()

            running_loss = running_loss/len_dl
            r_y = r_y/len_dl
            r_dydx = r_dydx/len_dl
            
            r_y_no_scaled = r_y_no_scaled/len_dl
            r_dydx_no_scaled = r_dydx_no_scaled/len_dl
            running_loss_no_scaled = alpha * r_y_no_scaled + beta * r_dydx_no_scaled

            stats['train_loss'].append(running_loss)
            stats['train_yloss'].append(r_y)
            stats['train_dyloss'].append(r_dydx)

            if stats[metric][-1] < best_loss :
                best_loss = stats[metric][-1]
                counter = 1
                torch.save(model.state_dict(), tmp_best_model_path)
            else :
                counter += 1
            
            print('Epoch {}: train loss: {}, y loss : {}, dy loss : {}'.format(epoch, running_loss, r_y, r_dydx))
            print()
            print('train loss no scaled: {}, y loss  no scaled : {}, dy loss  no scaled: {}'.format(
                  running_loss_no_scaled, r_y_no_scaled, r_dydx_no_scaled
            ))
            print()
            print()

            if counter == improving_limit + 1:
                break

    else :
        stats['train_loss'] = []

        for epoch in tqdm_notebook(range(max_epoch), desc=description):
            if learning_rate_schedule :
                # interpolate learning rate in cycle
                learning_rate = np.interp(epoch / max_epoch, lr_schedule_epochs, lr_schedule_rates)
                optimizer.lr = learning_rate
            
            running_loss = 0
            running_loss_no_scaled = 0

            for batch in dataloader:
              
                x, y = batch
                #x = x.to(device) 
        
                optimizer.zero_grad()

                # Forward pass
                y_pred = forward(net = model.net, x = x)

                # Compute Loss
                loss = criterion(y_pred.squeeze(), y)
                
                running_loss += loss.item()

                y_pred = y_mean + y_std * y_pred
                y = y_mean + y_std * y
                loss_no_scaled = criterion(y_pred.squeeze(), y)
                running_loss_no_scaled += loss_no_scaled.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            running_loss = running_loss/len_dl
            running_loss_no_scaled = running_loss_no_scaled/len_dl

            stats[metric].append(running_loss)

            if stats[metric][-1] < best_loss :
                best_loss = stats[metric][-1]
                counter = 1
                torch.save(model.state_dict(), tmp_best_model_path)
            else :
                counter += 1
            
            print('Epoch {}: train loss: {} train loss no scaled: {}'.format(epoch, running_loss, running_loss_no_scaled))
            print()

            if counter == improving_limit + 1 :
                break
    
    # load best model parameters
    model.load_state_dict(torch.load(tmp_best_model_path))
    os.remove(tmp_best_model_path)
    
    return model, stats, best_loss
    
def test(name, model, dataloader, criterion, config, with_derivative):

    assert name in ["net", "twin_net"]
    assert all([model, dataloader, criterion])

    description = config.get("description", "test...")
    x_mean, x_std = config.get("x_mean", 0.), config.get("x_std", 1.)
    y_mean, y_std = config.get("y_mean", 0.), config.get("y_std", 1.)
    dydx_mean, dydx_std = config.get("dydx_mean", 0.), config.get("dydx_std", 1.)
    alpha, beta = config.get("alpha", 1.), config.get("beta", 1.)
    
    len_dl = len(dataloader)
    assert len_dl

    model.eval()

    if with_derivative :
        running_loss, r_y, r_dydx = 0, 0, 0
        r_y_no_scaled, r_dydx_no_scaled = 0, 0

        x_list = []
        y_list = []
        dydx_list = []
        y_pred_list = []
        dydx_pred_list = []
        
        for batch in tqdm_notebook(dataloader, desc=description) :
              
            x, y, dydx = batch
            x_scaled = (x-x_mean) / x_std
            y_scaled = (y-y_mean) / y_std
            dydx_scaled = (dydx - dydx_mean) / dydx_std

            if name == "net" :
                # Forward pass
                x_scaled.requires_grad_(True)
                try :
                    y_pred_scaled = forward(net = model.net, x = x_scaled)
                except RuntimeError : # Expected object of scalar type Double but got scalar type Float for argument #3 'mat2' in call to _th_addmm_out
                    y_pred_scaled = forward(net = model.net, x = x_scaled.float())
                # Compute gradient 
                dydx_pred_scaled = gradient(y_pred_scaled, x_scaled)
            else :
                # Forward pass
                try :
                    y_pred_scaled, zs = forward(net = model.net, x = x_scaled, return_layers = True)
                except RuntimeError : # RuntimeError: Expected object of scalar type Double but got scalar type Float for argument #3 'mat2' in call to _th_addmm_out
                    y_pred_scaled, zs = forward(net = model.net, x = x_scaled.float(), return_layers = True)

                # Compute gradient 
                dydx_pred_scaled = backprop(net = model.net, y = y_pred_scaled, zs = zs)
                
            
            y_pred = y_mean + y_std * y_pred_scaled
            dydx_pred = dydx_mean + dydx_std * dydx_pred_scaled

            # Compute Loss
            l_y = criterion(y_pred_scaled.squeeze(), y_scaled)
            l_dydx = criterion(dydx_scaled, dydx_pred_scaled.detach())

            loss = alpha * l_y + beta * l_dydx
                
            running_loss += loss.item()
            r_y += l_y.item()
            r_dydx += l_dydx.item()

            l_y_no_scaled = criterion(y_pred.squeeze(), y).item()
            l_dydx_no_scaled = criterion(dydx , dydx_pred).item()
            r_y_no_scaled += l_y_no_scaled
            r_dydx_no_scaled += l_dydx_no_scaled
            
            x_list.append(x.data.numpy())
            y_list.append(y.data.numpy())
            dydx_list.append(dydx.data.numpy())
            y_pred_list.append(y_pred.data.numpy())
            dydx_pred_list.append(dydx_pred.data.numpy())

        running_loss = running_loss/len_dl
        r_y = r_y/len_dl
        r_dydx = r_dydx/len_dl

        r_y_no_scaled = r_y_no_scaled / len_dl
        r_dydx_no_scaled = r_dydx_no_scaled / len_dl
        running_loss_no_scaled = alpha * r_y_no_scaled + beta * r_dydx_no_scaled
            
        print('test loss: {}, y loss : {}, dydx loss : {}'.format(running_loss, r_y, r_dydx))
        print()
        print('test loss no scaled: {}, y loss  no scaled : {}, dydx loss  no scaled: {}'.format(
            running_loss_no_scaled, r_y_no_scaled, r_dydx_no_scaled
        ))
        print()
        print()

        x_list = list(itertools.chain.from_iterable(x_list))
        y_list = list(itertools.chain.from_iterable(y_list))
        dydx_list = list(itertools.chain.from_iterable(dydx_list))
        y_pred_list = list(itertools.chain.from_iterable(y_pred_list))
        try :
            dydx_pred_list = list(itertools.chain.from_iterable(dydx_pred_list))
        except TypeError : # iteration over a 0-d array
            pass

        return (running_loss, r_y, r_dydx), (x_list, y_list, dydx_list, y_pred_list, dydx_pred_list)

    else :
  
        running_loss = 0
        running_loss_no_scaled = 0

        x_list = []
        y_list = []
        y_pred_list = []
      
        for batch in tqdm_notebook(dataloader, desc=description) :
              
            x, y = batch
            x_scaled = (x-x_mean) / x_std
            y_scaled = (y-y_mean) / y_std
        
            # Forward pass
            y_pred_scaled = forward(net = model.net, x = x_scaled.float())
            y_pred = y_mean + y_std * y_pred_scaled

            # Compute Loss
            
            loss = criterion(y_pred_scaled.squeeze(), y_scaled)
                
            running_loss += loss.item()

            loss_no_scaled = criterion(y_pred.squeeze(), y)
            running_loss_no_scaled += loss_no_scaled.item()

            x_list.append(x.data.numpy())
            y_list.append(y.data.numpy())
            y_pred_list.append(y_pred.data.numpy())
                
        running_loss = running_loss/len_dl
        running_loss_no_scaled = running_loss_no_scaled/len_dl

        print('test loss: {} test loss no scaled {}'.format(running_loss, running_loss_no_scaled))

        x_list = list(itertools.chain.from_iterable(x_list))
        y_list = list(itertools.chain.from_iterable(y_list))
        y_pred_list = list(itertools.chain.from_iterable(y_pred_list))

        return (running_loss, None, None), (x_list, y_list, None, y_pred_list, None)


keys1 = ["normal_training", "sobolev_training", "twin_net_tf_differential", "twin_net_tf_normal" ,"twin_net_pytorch"]
keys2 = ["mlp", "siren"]
keys3 = ["no_normalize", "normalize"]
keys4 = [0, 1]
keys5 = ['train_yloss', 'train_dyloss']

def global_stat(stats_dic, suptitle = ""):
    global keys1, keys2, keys3, keys4, keys5

    # keys4 keys3 keys5 keys1 keys2

    for key4 in keys4 :
        fig, ax = plt.subplots(2, 2, sharex=False, figsize = (20, 8))
        fig.suptitle(key4)
        for i, key3 in enumerate(keys3) :
            for j, key5 in enumerate(keys5) :
                for key1 in keys1 :
                    for key2 in keys2 :
                        try :
                            y = stats_dic[key1][key2][key3][key4][key5]
                            x = range(len(y))
                            ax[i][j].plot(x, y, label = key1 +"-"+key2)
                        except (KeyError, TypeError) : # 'train_yloss', 'NoneType' object is not subscriptable
                              pass
                        
                    ax[i][j].set(xlabel = 'epoch' if i != 0 else "", ylabel = key5)
                    ax[i][j].set_title('%s per epoch' % key5 if i != 1 else "")
                    ax[i][j].legend()
                    #ax[i][j].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

def to_csv(dico, csv_path, n_samples : str = "", mode : str = 'a+'):
    global keys1, keys2, keys3, keys4, keys5
    # keys4 keys3 keys1 keys2
    rows = []
    result = {}
    for key4 in keys4 :
        result[key4] = {}
        for key3 in keys3 :
            key3_tmp = key3 + ('' if key4==0 else "-lr_scheduler") + ("-"+n_samples if n_samples else "")
            result[key4][key3_tmp] = {}
            for key1 in keys1 :
                for key2 in keys2 :
                    try :
                        key = key1 +"-"+key2
                        rows.append(key)
                        result[key4][key3_tmp][key] = dico[key1][key2][key3][key4]
                    except (KeyError, TypeError) : # 'train_yloss', 'NoneType' object is not subscriptable
                        result[key4][key3_tmp][key] = "RAS" 

    pd.DataFrame(result[0]).to_csv(csv_path, index = rows, mode = mode)
    pd.DataFrame(result[1]).to_csv(csv_path, index= rows, mode= mode)

    return list(set(rows)), result