import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import os 

from diff_ml_utils import normalize_data

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def genData(function, deriv_function, min_x, max_x, num_samples):
    fnd1 = deriv_function(index = 0)
    fnd2 = deriv_function(index = 1)
    samples = []
    for n in range(num_samples):
        x = np.array([random.uniform(min_x, max_x) for i in range(2)])
        y = function(x)
        dy1 = fnd1(x)
        dy2 = fnd2(x)
        dy = np.array([dy1, dy2])
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
        config["get_alpha_beta"] = lambda lam : (1.0, 1.0)
        
        return dataloader, config
        
    else :
        m = len(x)
        x = np.array(x)
        y = np.array(y)
        dydx = np.array(dydx) if dydx else None

        _, n = x.shape
        
        x_mean, x_std, x, y_mean, y_std, y, dydx, lambda_j = normalize_data(x, y, dydx, m)
        config = {"x_mean" : x_mean, "x_std" : x_std, "y_mean" : y_mean, "y_std" : y_std, "lambda_j" : lambda_j, "n" : n}
        def get_alpha_beta(lam) :
            alpha = 1.0 / (1.0 + lam * n)
            return alpha, 1.0 - alpha

        config["get_alpha_beta"] = get_alpha_beta 

        dataloader, _ = get_data_loader(x, y, dydx, batch_size, normalize = False)

        return dataloader, config
            
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_last = False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.is_last = is_last
        
    def forward(self, x):
        output = self.fc(x)
        if not self.is_last :
            output = F.relu(output)
        return output

class MLP_Relu(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        
        self.net = []

        # input layer
        self.net.append(Linear(in_features, hidden_features))
    
        # hidden layer(s)
        for i in range(hidden_layers):
            self.net.append(Linear(hidden_features, hidden_features))

        # output layer
        self.net.append(Linear(hidden_features, out_features, is_last = True))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(x)
        return output, x 

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True, is_first = False, omega_0=30, is_last = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.is_last :
            return self.omega_0 * self.linear(input)
        else :
            return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0, is_last = True))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def train(model, dataloader, optimizer, criterion, config = {'alpha':1, "beta" : 1}, 
          with_derivative = False, max_epoch = 20, improving_limit = 10, metric = "train_loss"):

    alpha = config["alpha"]
    beta = config["beta"]
    
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
                l_y = criterion(y_pred.squeeze(), y)
                l_dy = criterion(dy, dy_pred)

                loss = alpha * l_y + beta * l_dy
                
                running_loss += (l_y + l_dy).item()
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
    
def test(model, dataloader, criterion, with_derivative = False):
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
                
            # Forward pass
            y_pred, x = model(x)

            # Compute gradient
            dy_pred = gradient(y_pred, x)

            # Compute Loss
            l_y = criterion(y_pred.squeeze(), y)
            l_dy = criterion(dy, dy_pred)

            loss = l_y + l_dy
                
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
            #x = x.to(device) 
        
            # Forward pass
            y_pred, _ = model(x)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y)
                
            running_loss += loss.item()

            y_list.append(y)
            y_pred_list.append(y_pred)
                
        running_loss = running_loss/len_dl

        print('test loss: {}'.format(running_loss))

        return running_loss, (y_list, y_pred_list)
        
        
def plot_stat(stats, with_derivative = False):
    if with_derivative :
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize = (20, 3))
        fig.suptitle('')

        ax1.plot(range(len(stats['train_loss'])), stats['train_loss'], label='train')
        ax1.set(xlabel='epoch', ylabel='loss')
        ax1.set_title('loss per epoch')
        ax1.legend()
        ax1.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax2.plot(range(len(stats['train_yloss'])), stats['train_yloss'], label='train')
        ax2.set(xlabel='epoch', ylabel='yloss')
        ax2.set_title('yloss per epoch')
        ax2.legend()
        ax2.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax3.plot(range(len(stats['train_dyloss'])), stats['train_dyloss'], label='train')
        ax3.set(xlabel='epoch', ylabel='dyloss')
        ax3.set_title('dyloss per epoch')
        ax3.legend()
        ax3.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

    else :
        plt.plot(range(len(stats['train_loss'])), stats['train_loss'])