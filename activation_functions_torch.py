import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F

#from tqdm import tqdm_notebook
import tqdm

from models import TransformerEncoderLayer, TransformerEncoder, Dense

######## model

class Transformer(nn.Module):

    def __init__(self, x1_dim, x2_dim, output_dim = 2, d_model: int = 512, num_heads: int = 8, d_k = None, d_v = None, 
                 num_encoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: str = "relu") -> None:
        super(Transformer, self).__init__()

        d_k = d_k if d_k is not None else d_model 
        d_v = d_v if d_v is not None else d_model
        self.d_model = d_model 
        linear_layer_activation = None
        self.w0 = Dense(x1_dim, d_model, activation = linear_layer_activation) # Projects client information to R^d_model
        #nn.init.xavier_uniform_(self.w0.weight)
        self.drop0 = nn.Dropout(dropout)
        self.w1 = Dense(x2_dim, d_model, activation = linear_layer_activation)
        #nn.init.xavier_uniform_(self.w1.weight)
        self.drop1 = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_k, d_v, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, dropout = dropout)

        self.classifier = Dense(d_model, output_dim) # Classifier

        self._reset_parameters()

    def _get_mask(self, x1, x2, seq_lens):
        bs = x1.size(0)
        mask1 = torch.zeros(bs, 1).to(torch.bool) # mask for x1 : False
        mask2 = []
        for i, x in zip(range(bs), x2) :
            sl = seq_lens[i]
            pad_len = x.size(0)-sl
            mask2.append(
                # False + True
                torch.cat([torch.zeros(sl).to(torch.bool), torch.ones(pad_len).to(torch.bool)], 0)
            )
        mask2 = torch.stack(mask2) # mask for x2 
        return torch.cat([mask1, mask2], dim=1).to(x1.device) # combined mask

    def forward(self, x1, x2, seq_lens = None, softmax = True):
        """
        x1 : (batch_size, x1_dim)
        x2 : (seq_len, batch_size, x2_dim)
        """
        x1_bar = self.drop0(self.w0(x1)) # (batch_size, d_model)
        x1_bar = x1_bar.unsqueeze(1) # (batch_size, 1, d_model)
        output = self.drop1(self.w1(x2)) # (batch_size, seq_len, d_model)
        output = torch.cat([x1_bar, output], dim=1) # (batch_size, seq_len+1, d_model)

        src_mask = None
        src_key_padding_mask = self._get_mask(x1, x2, seq_lens) if seq_lens is not None else None
        #src_key_padding_mask = None
        output, _ = self.encoder(output, mask=src_mask, src_key_padding_mask=src_key_padding_mask) # (batch_size, seq_len+1, d_model)
        C = output[:, 0] # (batch_size, d_model)
        logits = self.classifier(C) # (batch_size, n_label)
        if softmax :
            logits = F.softmax(logits, dim=-1)
        return logits

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class RNN(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, output_dim, dropout= 0.1, variant="RNN", num_layers=1, bidirectional = False):
        super().__init__()
        assert variant in ["RNN", "LSTM"]
        self.w1 = nn.Linear(x1_dim, hidden_dim)
        #nn.init.xavier_uniform_(self.w1.weight)
        self.drop1 = nn.Dropout(dropout)
        d = dropout
        if num_layers == 1 :
            d = 0
            self.drop2 = nn.Dropout(dropout)
        else :
            self.drop2 = nn.Dropout(0)
        self.num_layers = num_layers
        self.factor = 2 if bidirectional else 1
        if variant == "RNN":
            self.rnn = nn.RNN(x2_dim, hidden_dim, num_layers=num_layers, bidirectional = bidirectional, dropout = d)
        elif variant == "LSTM" :
            self.rnn  = nn.LSTM(x2_dim, hidden_dim, num_layers=num_layers, bidirectional = bidirectional, dropout = d)
        
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop3 = nn.Dropout(dropout)
        self.activ = nn.Tanh()
        
        self.classifier = nn.Linear(hidden_dim*self.factor, output_dim)

    def forward(self, x1, x2, seq_lens = None, softmax = True):
        """
        x1 : (batch_size, x1_dim)
        x2 : (seq_len, batch_size, x2_dim)
        """
        h_0 = self.drop1(self.w1(x1)) # (batch_size, hidden_dim)
        #h_0 = h_0.unsqueeze(0) # (1, batch_size, hidden_dim)
        h_0 = h_0.repeat(self.num_layers*self.factor, 1, 1)
        if isinstance(self.rnn, nn.LSTM) :
            c_0 = torch.zeros_like(h_0)
            nn.init.xavier_normal_(c_0)
            output, (hidden, _) = self.rnn(x2, (h_0, c_0)) # (seq_len, batch_size, hidden_dim), (1, batch_size, hidden_dim)
        else :
            output, hidden = self.rnn(x2, h_0) # (seq_len, batch_size, hidden_dim), (1, batch_size, hidden_dim)
        # todo : Bahdanau attention
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        if seq_lens is not None :
            # take the last hidden state before padded sequence
            batch_size = x1.size(0)
            hidden = torch.stack([output[seq_lens[i]-1,i,:] for i in range(batch_size)]) # (batch_size, hidden_dim)
        else :
            hidden = hidden.squeeze() # (batch_size, hidden_dim)
        hidden = self.drop3(hidden)
        hidden = self.activ(self.w2(hidden)) # (batch_size, hidden_dim)
        out = self.classifier(self.drop3(hidden))
        if softmax :  
            out = F.softmax(out, dim = -1)
        return out

class SigmoidModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m1 = model
        self.m2 = nn.Sigmoid()
    def forward(self, x1, x2, seq_lens):
        output = self.m1(x1, x2, seq_lens, softmax = False).squeeze()
        return F.sigmoid(output)

########## data ##############
def process_frame(client_df, invoice_df) :
    client_df['creation_day'] = client_df['creation_date'].apply(lambda date: int(date[:2]))
    client_df['creation_month'] = client_df['creation_date'].apply(lambda date: int(date[3:5]))
    client_df['creation_year'] = client_df['creation_date'].apply(lambda date: int(date[-4:]))
    client_df = client_df.sort_values(by=['client_id'], axis=0, ascending=True, inplace=False)
    client_df = client_df.drop(['creation_date'], axis=1)

    invoice_df['invoice_day'] = invoice_df['invoice_date'].apply(lambda date: int(date[-2:]))
    invoice_df['invoice_month'] = invoice_df['invoice_date'].apply(lambda date: int(date[5:7]))
    invoice_df['invoice_year'] = invoice_df['invoice_date'].apply(lambda date: int(date[0:4]))
    invoice_df = invoice_df.sort_values(by=['client_id', 'invoice_date'], axis=0, ascending=True, inplace=False)
    invoice_df = invoice_df.drop(['invoice_date'], axis=1)

    invoice_df['counter_type'] = invoice_df['counter_type'].apply(lambda x : 0 if x == "ELEC" else 1)
    invoice_df['counter_statue'] = invoice_df['counter_statue'].apply(lambda x : 0 if type(x) == str else x)

    return client_df, invoice_df

class FDDataset(Dataset):
    def __init__(self, client_df = None, invoice_df = None, data = None, sorted = True, reverse=False,
                 batch_size = 1): # data_frame and pipeline object
        super().__init__()

        self.data = data if data is not None else self.prepare_data(client_df, invoice_df)
        if sorted :
            self.data.sort(reverse=reverse, key = lambda x : x[1].size(0))
        
        self.batch_size = len(self.data) if batch_size > len(self.data) else batch_size

    def __len__(self):
        return len(self.data)//self.batch_size
    
    def __getitem__(self, index) :
        return self.data[index]

    def prepare_data(self, client_df, invoice_df):
        # client_id, district, client_catg, region, creation_date, ...
        client_df_columns = list(client_df.columns)
        # client_id, invoice_date, tariff_type, counter_number, counter_statue, counter_code, 
        # reading_remark, counter_coefficient, consumption_level_1, consumption_level_2, 
        # consumption_level_3, consumption_level_4, old_index, new_index, months_number, counter_type
        invoice_df_columns = list(invoice_df.columns)
        client_id_vc = invoice_df['client_id'].value_counts()

        data = []
        start = 0
        description = "loading data ..."
        for index in tqdm.notebook.tqdm(range(len(client_df)), desc=description):
            client = client_df.iloc[index]
            client_id = client["client_id"]
            target = client["target"]
            target = torch.tensor(target, dtype=torch.long)
            
            client = client.drop(["client_id",'target'])
            #x1 = [client[col] for col in client_df_columns if col != "client_id"]
            #x1 = [client.values[0]]
            #x1.extend(client.values[2:])
            x1 = client.values.astype(int)
            x1 = torch.tensor(x1, dtype=torch.float)

            
            #invoices = invoice_df[invoice_df['client_id'] == client_id]
            end = start + client_id_vc[client_id]
            invoices = invoice_df.iloc[start:end]
            start = end
            assert all((invoices['client_id'] == client_id).values)
            #x2 = [[invoices.iloc[i][col] for col in sinvoice_df_columns if col != "client_id"] for i in range(len(invoices))]
            x2 = invoices.values[:,1:].astype(int)
            try :
                x2 = torch.tensor(x2, dtype=torch.float)
            except TypeError :
                #x2 = torch.zeros(len(x2), len(x2[0])).to(torch.float)
                continue
                
            data.append((x1, x2, target)) # x1, x2, y

        return data

    def generate_batch(self, data_batch):
        """padding"""
        x1, x2, target = zip(*data_batch)
        max_len = max([x.size(0) for x in x2])
        seq_len = x2[0].size(-1)
        x2_temp = []
        seq_lens = []
        for x in x2 :
            seq_lens.append(x.size(0))
            pad_len = max_len-x.size(0)
            x2_temp.append(
                torch.cat([x, torch.zeros(pad_len, seq_len)], 0)
            )
        return torch.stack([x for x in x1]), torch.stack(x2_temp), torch.tensor(target), seq_lens

    def __iter__(self): # iterator to load data
       assert self.batch_size
       self.batch_size = len(self.data) if self.batch_size > len(self.data) else self.batch_size
       n_samples = len(self.data)
       i = 0
       while n_samples > i :
          i += self.batch_size
          yield self.generate_batch(self.data[i-self.batch_size:i])


class FDDataset4Test(FDDataset):
    def __init__(self, client_df = None, invoice_df = None, sorted = True, reverse=False,
                 batch_size = 1): # data_frame and pipeline object
        #super().__init__()

        self.data = self.prepare_data(client_df, invoice_df)
        if sorted :
            self.data.sort(reverse=reverse, key = lambda x : x[1].size(0))
        
        self.batch_size = len(self.data) if batch_size > len(self.data) else batch_size

    def prepare_data(self, client_df, invoice_df):
        client_df_columns = list(client_df.columns)
        invoice_df_columns = list(invoice_df.columns)
        client_id_vc = invoice_df['client_id'].value_counts()

        data = []
        start = 0
        description = "loading data ..."
        for index in tqdm.notebook.tqdm(range(len(client_df)), desc=description):
            client = client_df.iloc[index]
            x1 = [client.values[0]]
            x1.extend(client.values[2:])
            x1 = torch.tensor(x1, dtype=torch.float)

            client_id = client["client_id"]
            end = start + client_id_vc[client_id]
            invoices = invoice_df.iloc[start:end]
            start = end
            assert all((invoices['client_id'] == client_id).values)
            
            x2 = invoices.values[:,1:].astype(int)
            try :
                x2 = torch.tensor(x2, dtype=torch.float)
            except Exception as ex :
              #x2 = torch.zeros(len(x2), len(x2[0])).to(torch.float)
              # on doit charger tout les exemple de test
              #continue
              raise ex

            data.append((client_id, x1, x2)) # x1, x2, y

        return data

    def generate_batch(self, data_batch):
        """padding"""
        client_ids, x1, x2 = zip(*data_batch)
        max_len = max([x.size(0) for x in x2])
        seq_len = x2[0].size(-1)
        x2_temp = []
        seq_lens = []
        for x in x2 :
            seq_lens.append(x.size(0))
            pad_len = max_len-x.size(0)
            x2_temp.append(
                torch.cat([x, torch.zeros(pad_len, seq_len)], 0)
            )
        return client_ids, torch.stack([x for x in x1]), torch.stack(x2_temp), seq_lens

    def run_test(self, model, device, csv_file, type_ = 0):
        model.eval()
        prob_list = [] # list of probability
        y_pred_list = []
        client_ids = []
        description = "test step ..."

        permute_x2 = True
        #if isinstance(model, Transformer) :
        if str(type(model)).split(".")[-1] == "Transformer'>" :
            permute_x2 = False
        #if isinstance(model, SigmoidModel):
        if str(type(model)).split(".")[-1] == "SigmoidModel'>" :
            #if isinstance(model.m1, Transformer):
            if str(type(model.m1)).split(".")[-1] == "Transformer'>" :
                permute_x2 = False

        for batch in tqdm.notebook.tqdm(self, desc=description) :
            ids, x1, x2, seq_lens = batch
            client_ids.extend(ids)
            x1 = x1.to(device)
            x2 = x2.to(device)
            if permute_x2 :
                x2 = x2.contiguous().permute(1, 0, 2) # (seq_len, batch_size, _)
            logits = model(x1, x2, seq_lens)

            if type_ == 0 : # non sigmoid
                prob, label_pred = logits.max(1)
                y_pred_list.extend(label_pred.cpu().numpy())
                prob_list.extend(prob.cpu().detach().numpy())
            else :
                #y_pred_list.extend((logits>0.5).to(int).cpu().numpy())
                y_pred_list.extend((logits<0.5).to(int).cpu().numpy())
                prob_list.extend(logits.cpu().detach().numpy())
      
        pd.DataFrame(zip(client_ids, prob_list)).to_csv(csv_file, header= ["client_id", "target"])
        
        self.prob_list = prob_list
        self.y_pred_list = y_pred_list

##### training

def train_step(model, optimizer, criterion, data, device, permute_x2 = True, description = "train step ...", type_ = 0):
    model.train()
    total_loss = 0
    y_list = []
    y_pred_list = []
    l = len(data)
    for batch in tqdm.notebook.tqdm(data, desc=description) :
        x1, x2, y, seq_lens = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if permute_x2 :
          x2 = x2.contiguous().permute(1, 0, 2) # (seq_len, batch_size, _)
        logits = model(x1, x2, seq_lens)
        try :
            loss = criterion(logits, y)
        except :
            y = y.to(torch.float)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        #x1 = x1.cpu()
        #x2 = x2.cpu()
        #y = y.cpu()

        total_loss += loss.item()
        if type_ == 0 : # non sigmoid
            y_list.extend(y.cpu().numpy())
            _, label_pred = logits.max(1)
            y_pred_list.extend(label_pred.cpu().numpy())
        else :
            y_list.extend(y.cpu().numpy())
            y_pred_list.extend((logits>0.5).to(int).cpu().numpy())
    
    return total_loss/l, y_list, y_pred_list


def evaluate(model, criterion, data, device, permute_x2 = True, description = "eval step ...", type_ = 0):
    model.eval()
    total_loss = 0
    y_list = []
    y_pred_list = []
    l = len(data)
    for batch in tqdm.notebook.tqdm(data, desc=description) :
        x1, x2, y, seq_lens = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        if permute_x2 :
            x2 = x2.contiguous().permute(1, 0, 2) # (seq_len, batch_size, _)
        logits = model(x1, x2, seq_lens)
        try :
            loss = criterion(logits, y)
        except :
            y = y.to(torch.float)
            loss = criterion(logits, y)

        total_loss += loss.item()

        #x1 = x1.cpu()
        #x2 = x2.cpu()
        #y = y.cpu()

        if type_ == 0 : # non sigmoid
            y_list.extend(y.cpu().numpy())
            _, label_pred = logits.max(1)
            y_pred_list.extend(label_pred.cpu().numpy())
        else :
            y_list.extend(y.cpu().numpy())
            y_pred_list.extend((logits>0.5).to(int).cpu().numpy())
    
    return total_loss/l, y_list, y_pred_list

def train(model, optimizer, criterion, train_data, val_data, device, n_epochs, type_ = 0, save_path = "./model.pth") :
    best_score = 0
    permute_x2 = True
    #if isinstance(model, Transformer) :
    if str(type(model)).split(".")[-1] == "Transformer'>" :
        permute_x2 = False
    #if isinstance(model, SigmoidModel):
    if str(type(model)).split(".")[-1] == "SigmoidModel'>" :
        #if isinstance(model.m1, Transformer):
        if str(type(model.m1)).split(".")[-1] == "Transformer'>" :
            permute_x2 = False

    for i in range(n_epochs):
        description = "epoch %d"%i
        train_loss, y_list, y_pred_list = train_step(model, optimizer, criterion, train_data, device, permute_x2, description, type_)
        train_fs = f1_score(y_list, y_pred_list)
        train_acc = accuracy_score(y_list, y_pred_list)
        print("train -> loss : {}, acc : {}, f1-score : {}".format(train_loss, train_acc, train_fs))

        
        if val_data is not None :
            description = "val step"
            val_loss, y_list, y_pred_list = evaluate(model, criterion, val_data, device, permute_x2, description, type_)
            val_fs = f1_score(y_list, y_pred_list)
            val_acc = accuracy_score(y_list, y_pred_list)
            if val_fs > best_score :
                torch.save(model.state_dict(), save_path)
                best_score = val_fs

            print("val -> loss : {}, acc : {}, f1-score : {}".format(val_loss, val_acc, val_fs))
    try :
        model.load_state_dict(torch.load(save_path)) 
    except :
        pass 
    return model

def setting(model_class, lr = 3e-5, type_ = 0, model_kwargs = {}) :
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if type_ == 0 :
        criterion = nn.CrossEntropyLoss()
        model = model_class(**model_kwargs)
    else :
        criterion = nn.BCELoss()
        model_kwargs["output_dim"] = 1
        model = SigmoidModel(model_class(**model_kwargs))
    
    optimizer = Adam(model.parameters(), lr=lr)

    model = model.to(device)
    criterion = criterion.to(device)
    if n_gpu > 1 : # use Data Parallelism with Multi-GPU  
        model = nn.DataParallel(model)
    
    return model, optimizer, criterion, device


def get_upsampled(train_data):
    # separate minority and majority classes
    not_fraud = []
    fraud = []
    for x in train_data :
        if x[2] == 0 :
            not_fraud.append(x)
        else :
            fraud.append(x)


    # upsample minority
    fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results
      # combine majority and upsampled minority
    upsampled = not_fraud +  fraud_upsampled # this become a training data
    return upsampled

def get_undersampled(train_data):
    # separate minority and majority classes
    not_fraud = []
    fraud = []
    for x in train_data :
        if x[2] == 0 :
            not_fraud.append(x)
        else :
            fraud.append(x)
      
    # downsample majority
    not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 27) # reproducible results
    # combine minority and downsampled majority
    downsampled = not_fraud_downsampled + fraud

    return downsampled
