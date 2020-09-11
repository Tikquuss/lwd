# Please refer to the following link for implementation details.
# https://github.com/differential-machine-learning/notebooks/blob/master/DifferentialML.ipynb

import torch
from torch import nn

import numpy as np
import os

from utils import gradient

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias = True, omega_0 = 1, activation_function = None, deriv_activation_function = None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation_function = activation_function if activation_function else lambda x : x
        self.deriv_activation_function = deriv_activation_function if deriv_activation_function else lambda x : x
        self.omega_0 = omega_0
  
    def forward(self, x):
        # assert torch.all(torch.eq(self.linear(x), torch.matmul(x, self.linear.weight.t()) + self.linear.bias))
        z = self.omega_0 * self.linear(x)
        y = self.activation_function(z)
        return y, z

class SineLayer(Linear):
    def __init__(self, in_features, out_features, bias = True, is_first = False, is_last = False, omega_0 = 30):
        
        if is_last :
            activation_function = None
            deriv_activation_function = None
        else :
            activation_function = torch.sin
            deriv_activation_function = torch.cos

        super(SineLayer, self).__init__(in_features, out_features, bias = bias, 
                                        omega_0 = omega_0, activation_function = activation_function, 
                                        deriv_activation_function = deriv_activation_function)
        
        self.in_features = in_features
        self.is_first = is_first 
        self.is_last = is_last

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
                
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear = False, 
                 first_omega_0=30, hidden_omega_0=30.):

        super(Siren, self).__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0 = first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0 = hidden_omega_0))

        if outermost_linear:
            final_linear = Linear(hidden_features, out_features, omega_0 = 1)
            
            with torch.no_grad():
                final_linear.linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0, is_last = True))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x, return_x = False):
        if return_x :
            x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            for siren_layer in self.net :
                 x, _ = siren_layer(x)
            y = x
            return y, x_grad
        else :
            zs = []
            x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            for linear_layer in self.net :
                 x, z = linear_layer(x)
                 zs.append(z)
            y = x
            return y, zs 

    def forward_only(self, x):
        x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        for siren_layer in self.net :
             x, _ = siren_layer(x)
        y = x
        return y, x_grad
    
    def backprop(self, y, zs):
       
        zbar = torch.ones_like(y)
        for l in range(len(self.net) - 1, 0, -1):
            siren_layer = self.net[l] 
            zbar =  siren_layer.omega_0 * torch.matmul(zbar, siren_layer.linear.weight) * siren_layer.deriv_activation_function(zs[l-1])

            # Thanks https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
            zbar.detach_() # eq. zbar = zbar.detach()

        siren_layer = self.net[0]
        zbar = siren_layer.omega_0 * torch.matmul(zbar, siren_layer.linear.weight)

        # Thanks https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
        zbar.detach_() # eq. zbar = zbar.detach()
        
        xbar = zbar        
      
        return xbar
        

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, activation_function = None, deriv_activation_function = None):
        super().__init__()
        
        self.net = []

        # input layer
        self.net.append(Linear(in_features, hidden_features, activation_function, deriv_activation_function = deriv_activation_function))
    
        # hidden layer(s)
        for i in range(hidden_layers):
            self.net.append(Linear(hidden_features, hidden_features, activation_function, deriv_activation_function = deriv_activation_function))

        # output layer
        self.net.append(Linear(hidden_features, out_features,))

        self.net = nn.Sequential(*self.net)


    def forward(self, x, return_x = False):
        if return_x :
            x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            for siren_layer in self.net :
                 x, _ = siren_layer(x)
            y = x
            return y, x_grad
        else :
            zs = []
            x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            for linear_layer in self.net :
                 x, z = linear_layer(x)
                 zs.append(z)
            y = x
            return y, zs 

    def forward_only(self, x):
        x_grad = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        for siren_layer in self.net :
             x, _ = siren_layer(x)
        y = x
        return y, x_grad
    
    def backprop(self, y, zs):
        zbar = torch.ones_like(y)
        for l in range(len(self.net) - 1, 0, -1):
            linear_layer = self.net[l] 
            zbar =  torch.matmul(zbar, linear_layer.linear.weight) * linear_layer.deriv_activation_function(zs[l-1])
            # Thanks https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
            zbar.detach_() # eq. zbar = zbar.detach()

        linear_layer = self.net[0]
        zbar = torch.matmul(zbar, linear_layer.linear.weight)
        # Thanks https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
        zbar.detach_() # eq. zbar = zbar.detach()
        
        xbar = zbar

        # dz[L] / dx
        return xbar 

class TwinNet():
    def __init__(self, model, optimizer, criterion):
        assert all([model, optimizer, criterion])
        assert any([isinstance(model, className) for className in [Siren, MLP]]), "Model type not supported"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, dataloader, config, lam = 1,  with_derivative = False, max_epoch = 20, improving_limit = 10, metric = "train_loss"):

        x_std, y_mean, y_std = config["x_std"], config["y_mean"], config["y_std"]
        lambda_j = config["lambda_j"]
        alpha = 1.0 / (1.0 + lam * config["n"])
        beta = 1.0 - alpha

        assert dataloader

        if with_derivative :
            assert metric in ["train_loss", "train_yloss", "train_dyloss"]
        else :
            assert metric == "train_loss"

        len_dl = len(dataloader)
        assert len_dl

        self.model.train()
        
        tmp_best_model_path = './best-model-tmp.pth'
        best_loss = float('inf')
        counter = 1
        stats = {}
        
        if with_derivative :
            stats['train_loss'] = []
            stats['train_yloss'] = []
            stats['train_dyloss'] = []

            for epoch in range(max_epoch):

                running_loss = 0
                r_y, r_dy = 0, 0

                for batch in dataloader:
                  
                    x, y, dy = batch

                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    y_pred, zs = self.model(x)
                    
                    # Compute gradient
                    dy_pred = self.model.backprop(y_pred, zs) 

                    # Compute Loss
                    #y_pred = y_mean + y_std * y_pred
                    #dy_pred = y_std / x_std *  dy_pred

                    l_y = self.criterion(y_pred.squeeze(), y)
                    l_dy = self.criterion(dy * lambda_j, dy_pred * lambda_j)

                    loss = alpha * l_y + beta * l_dy
                    
                    running_loss += loss.item()
                    r_y += l_y.item()
                    r_dy += l_dy.item()
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                running_loss = running_loss/len_dl
                r_y = r_y/len_dl
                r_dy = r_dy/len_dl

                stats['train_loss'].append(running_loss)
                stats['train_yloss'].append(r_y)
                stats['train_dyloss'].append(r_dy)

                if stats[metric][-1] < best_loss :
                    best_loss = stats[metric][-1]
                    counter = 1
                    torch.save(self.model.state_dict(), tmp_best_model_path)
                else :
                    counter += 1
                
                print('Epoch {}: train loss: {}, y loss : {}, dy loss : {}'.format(epoch, running_loss, r_y, r_dy))

                if  counter == improving_limit + 1:
                    break

        else :
            stats['train_loss'] = []

            for epoch in range(max_epoch):
                
                running_loss = 0
                
                for batch in dataloader:
                  
                    x, y = batch
                    #x = x.to(device) 
            
                    self.optimizer.zero_grad()
                    # Forward pass
                    y_pred, _ = self.model(x)

                    # Compute Loss
                    #y_pred = y_mean + y_std * y_pred
                    loss = self.criterion(y_pred.squeeze(), y)
                    
                    running_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                running_loss = running_loss/len_dl

                stats[metric].append(running_loss)

                if stats[metric][-1] < best_loss :
                    best_loss = stats[metric][-1]
                    counter = 1
                    torch.save(self.model.state_dict(), tmp_best_model_path)
                else :
                    counter += 1
                
                print('Epoch {}: train loss: {}'.format(epoch, running_loss))

                if  counter == improving_limit + 1 :
                    break
        
        # load best model parameters
        self.model.load_state_dict(torch.load(tmp_best_model_path))
        os.remove(tmp_best_model_path)
        self.best_loss = best_loss
        self.stats = stats
    
    def test(self, dataloader, config, lam = 1, with_derivative = False):
        
        lambda_j = config["lambda_j"]
        alpha = 1.0 / (1.0 + lam * config["n"])
        beta = 1.0 - alpha
        x_mean, x_std, y_mean, y_std = config["x_mean"], config["x_std"], config["y_mean"], config["y_std"]

        assert dataloader
        
        len_dl = len(dataloader)
        assert len_dl

        self.model.eval()

        if with_derivative :
            running_loss, r_y, r_dy = 0, 0, 0

            y_list = []
            dy_list = []
            y_pred_list = []
            dy_pred_list = []
            
            for batch in dataloader:
                  
                x, y, dy = batch

                x_scaled = (x-x_mean) / x_std
                    
                # Forward pass
                y_pred_scaled, zs  = self.model(x_scaled.float())
                
                y_pred = y_mean + y_std * y_pred_scaled

                # Compute gradient
                dy_pred_scaled = self.model.backprop(y_pred_scaled, zs)

                dy_pred = y_std / x_std * dy_pred_scaled 

                # Compute Loss
                y_scaled = (y-y_mean) / y_std
                dy_scaled = dy / y_std * x_std 
                l_y = self.criterion(y_pred_scaled.squeeze(), y_scaled)
                l_dy = self.criterion(dy_scaled * lambda_j, dy_pred_scaled * lambda_j)

                loss = alpha * l_y + beta * l_dy
                    
                running_loss += loss.item()
                r_y += l_y.item()
                r_dy += l_dy.item()
                
                y_list.append(y)
                dy_list.append(dy)
                y_pred_list.append(y_pred)
                dy_pred_list.append(dy_pred)

            running_loss = running_loss/len_dl
            r_y = r_y/len_dl
            r_dy = r_dy/len_dl
                
            print('test loss: {}, y loss : {}, dy loss : {}'.format(running_loss, r_y, r_dy))

            return running_loss, r_y, r_dy, (y_list, dy_list, y_pred_list, dy_pred_list)

        else :
      
            running_loss = 0
            y_list = []
            y_pred_list = []
          
            for batch in dataloader:
                  
                x, y = batch
                x_scaled = (x-x_mean) / x_std
                #x = x.to(device) 
            
                # Forward pass
                y_pred_scaled, _ = self.model(x_scaled.float())

                y_pred = y_mean + y_std * y_pred_scaled

                # Compute Loss
                y_scaled = (y-y_mean) / y_std
                loss = self.criterion(y_pred_scaled.squeeze(), y_scaled)
                    
                running_loss += loss.item()

                y_list.append(y)
                y_pred_list.append(y_pred)
                    
            running_loss = running_loss/len_dl

            print('test loss: {}'.format(running_loss))

            return running_loss, (y_list, y_pred_list)


def train(model, dataloader, optimizer, criterion, config, lam = 1, with_derivative = False, max_epoch = 20, improving_limit = 10, metric = "train_loss"):
    
    x_std, y_mean, y_std = config["x_std"], config["y_mean"], config["y_std"]
    lambda_j = config["lambda_j"]
    alpha = 1.0 / (1.0 + lam * config["n"])
    beta = 1.0 - alpha


    assert all([model, dataloader, criterion])

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

        for epoch in range(max_epoch):

            running_loss = 0
            r_y, r_dy = 0, 0

            for batch in dataloader:
              
                x, y, dy = batch

                optimizer.zero_grad()
                
                # Forward pass
                y_pred, x = model(x)
                
                # Compute gradient
                dy_pred = gradient(y_pred, x)

                # Compute Loss
                #y_pred = y_mean + y_std * y_pred
                #dy_pred = y_std / x_std *  dy_pred
                l_y = criterion(y_pred.squeeze(), y)
                l_dy = criterion(dy * lambda_j, dy_pred.detach() * lambda_j)

                loss = alpha * l_y + beta * l_dy
                
                running_loss += loss.item()
                r_y += l_y.item()
                r_dy += l_dy.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()

            running_loss = running_loss/len_dl
            r_y = r_y/len_dl
            r_dy = r_dy/len_dl

            stats['train_loss'].append(running_loss)
            stats['train_yloss'].append(r_y)
            stats['train_dyloss'].append(r_dy)

            if stats[metric][-1] < best_loss :
                best_loss = stats[metric][-1]
                counter = 1
                torch.save(model.state_dict(), tmp_best_model_path)
            else :
                counter += 1
            
            print('Epoch {}: train loss: {}, y loss : {}, dy loss : {}'.format(epoch, running_loss, r_y, r_dy))

            if  counter == improving_limit + 1:
                break

    else :
        stats['train_loss'] = []

        for epoch in range(max_epoch):
            
            running_loss = 0
            
            for batch in dataloader:
              
                x, y = batch
                #x = x.to(device) 
        
                optimizer.zero_grad()
                # Forward pass
                y_pred, _ = model(x)

                # Compute Loss
                #y_pred = y_mean + y_std * y_pred
                loss = criterion(y_pred.squeeze(), y)
                
                running_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            running_loss = running_loss/len_dl

            stats[metric].append(running_loss)

            if stats[metric][-1] < best_loss :
                best_loss = stats[metric][-1]
                counter = 1
                torch.save(model.state_dict(), tmp_best_model_path)
            else :
                counter += 1
            
            print('Epoch {}: train loss: {}'.format(epoch, running_loss))

            if  counter == improving_limit + 1 :
                break
    
    # load best model parameters
    model.load_state_dict(torch.load(tmp_best_model_path))
    os.remove(tmp_best_model_path)
    
    return model, stats, best_loss
    
def test(model, dataloader, criterion, config, lam = 1, with_derivative = False):
    lambda_j = config["lambda_j"]
    alpha = 1.0 / (1.0 + lam * config["n"])
    beta = 1.0 - alpha
    x_mean, x_std, y_mean, y_std = config["x_mean"], config["x_std"], config["y_mean"], config["y_std"]

    assert all([model, dataloader, criterion])
    
    len_dl = len(dataloader)
    assert len_dl

    if with_derivative :
        model.train()
    else :
        model.eval()

    if with_derivative :
        running_loss, r_y, r_dy = 0, 0, 0

        y_list = []
        dy_list = []
        y_pred_list = []
        dy_pred_list = []
        
        for batch in dataloader:
              
            x, y, dy = batch
            x_scaled = (x-x_mean) / x_std
                
            # Forward pass
            y_pred_scaled, x_scaled = model(x_scaled.float())
            y_pred = y_mean + y_std * y_pred_scaled

            # Compute gradient
            dy_pred_scaled = gradient(y_pred_scaled, x_scaled)

            dy_pred = y_std / x_std * dy_pred_scaled 

            # Compute Loss
            y_scaled = (y-y_mean) / y_std
            dy_scaled = dy / y_std * x_std  
            l_y = criterion(y_pred_scaled.squeeze(), y_scaled)
            l_dy = criterion(dy_scaled * lambda_j, dy_pred_scaled.detach() * lambda_j)

            loss = alpha * l_y + beta * l_dy
                
            running_loss += loss.item()
            r_y += l_y.item()
            r_dy += l_dy.item()
            
            y_list.append(y)
            dy_list.append(dy)
            y_pred_list.append(y_pred)
            dy_pred_list.append(dy_pred)

        running_loss = running_loss/len_dl
        r_y = r_y/len_dl
        r_dy = r_dy/len_dl
            
        print('test loss: {}, y loss : {}, dy loss : {}'.format(running_loss, r_y, r_dy))

        return running_loss, r_y, r_dy, (y_list, dy_list, y_pred_list, dy_pred_list)

    else :
  
        running_loss = 0
        y_list = []
        y_pred_list = []
      
        for batch in dataloader:
              
            x, y = batch
            x_scaled = (x-x_mean) / x_std
            #x = x.to(device) 
        
            # Forward pass
            y_pred_scaled, _ = model(x_scaled.float())
            y_pred = y_mean + y_std * y_pred_scaled

            # Compute Loss
            y_scaled = (y-y_mean) / y_std
            loss = criterion(y_pred_scaled.squeeze(), y_scaled)
                
            running_loss += loss.item()

            y_list.append(y)
            y_pred_list.append(y_pred)
                
        running_loss = running_loss/len_dl

        print('test loss: {}'.format(running_loss))

        return running_loss, (y_list, y_pred_list)