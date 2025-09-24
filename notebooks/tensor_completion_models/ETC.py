import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys

from notebooks.tensor_completion_models.codes_costco_tucker.costco import *
from notebooks.tensor_completion_models.codes_costco_tucker.read import *
from notebooks.tensor_completion_models.codes_costco_tucker.utils import *

from notebooks.tensor_completion_models.CoSTCo import *
from notebooks.tensor_completion_models.CPD import *
from notebooks.utilities.helper_functions import *

from sklearn.utils.validation import check_random_state
import tensorly as tl

from sklearn.model_selection import train_test_split, KFold




class ETC(nn.Module):

    def __init__(self, 
                 sparse_tensor,
                 models_type_rank = ['cpd_10', 'cpd_15', 'costco_10', 'costco_15'],
                 num_splits = 5,
                 use_unique_train_values = False,
                 use_all_train_values = False,
                 agg_func = 'median',
                 mlp_hidden_dim1 = 100,
                 mlp_hidden_dim2 = None,
                 dropout_p = 0.0,
                 dropout_p_before_mlp = 0.0,
                 cnn_hidden_channels1 = 32,
                 cnn_hidden_channels2 = None,
                 tucker_in_drop=0.1,
                 tucker_hidden_drop=0.1,
                 lr = 5e-3,
                 wd = 1e-4,
                 num_epochs = 7500,
                 batch_size = 128,
                 further_train_individuals = False,
                 for_queries = False,
                 device = "cuda" if torch.cuda.is_available() else "cpu"):
        
        super(ETC, self).__init__()

        self.sparse_tensor = sparse_tensor
        self.mlp_hidden_dim1 = mlp_hidden_dim1
        self.mlp_hidden_dim2 = mlp_hidden_dim2
        self.cnn_hidden_channels1 = cnn_hidden_channels1
        self.cnn_hidden_channels2 = cnn_hidden_channels2
        self.agg_func = agg_func

        self.models_type_rank = models_type_rank
        
        self.models = list()
        
        if (num_splits is None) or (use_all_train_values):
            kf = KFold(n_splits=len(self.models_type_rank), shuffle = True, random_state = 18)
        elif (num_splits > (self.sparse_tensor.indices().t().shape[0])):
            kf = KFold(n_splits=(self.sparse_tensor.indices().t().shape[0]), shuffle = True, random_state = 18)
        elif (use_unique_train_values) or (num_splits <= len(self.models_type_rank)):
            kf = KFold(n_splits=len(self.models_type_rank), shuffle = True, random_state = 18)
        else:                
            kf = KFold(n_splits=num_splits, shuffle = True, random_state = 18)
        
        cond = True
        model_i = 0
        while (cond):
            
            for i, (train_index, test_index) in enumerate(kf.split(self.sparse_tensor.indices().t())):
                
                if (model_i >= len(self.models_type_rank)): 
                    cond = False
                    break
                
                # print(model_i)
                    
                if (use_all_train_values):
                    current_indices = self.sparse_tensor.indices().t()
                    current_values = self.sparse_tensor.values() 
                elif (use_unique_train_values):
                    current_indices = self.sparse_tensor.indices().t()[test_index]
                    current_values = self.sparse_tensor.values()[test_index]
                else:
                    current_indices = self.sparse_tensor.indices().t()[train_index]
                    current_values = self.sparse_tensor.values()[train_index]
                    
                                        
                model_type_rank = self.models_type_rank[model_i]
                
                split_model_type_rank = tuple(model_type_rank.split('_'))

                model_type = split_model_type_rank[0]
                
                if (len(split_model_type_rank) > 1):
                    rank = int(split_model_type_rank[1])
                else: rank = 5
                
                NeAT_hidden_dim = 32                 
                if (model_type == 'NeAT' and len(split_model_type_rank) > 2): 
                    NeAT_hidden_dim = int(split_model_type_rank[2])    
                        
                        
                cpd_smooth_lambda = 2
                if (model_type == 'cpd.smooth' and len(split_model_type_rank) > 2):
                    cpd_smooth_lambda = float(split_model_type_rank[2])  
                    
                
                del split_model_type_rank
                            
                new_model = train_tensor_completion(model_type = model_type,
                                                    sparse_tensor = self.sparse_tensor,
                                                    rank = int(rank),
                                                    num_epochs = num_epochs,
                                                    batch_size = batch_size,
                                                    lr = lr,
                                                    wd = wd,
                                                    cpd_smooth_lambda = cpd_smooth_lambda,
                                                    NeAT_hidden_dim = NeAT_hidden_dim,
                                                    tucker_in_drop = tucker_in_drop,
                                                    tucker_hidden_drop=tucker_hidden_drop,
                                                    early_stopping = True,
                                                    flags = 20,
                                                    verbose = False,
                                                    val_size = 0.2,
                                                    for_queries=for_queries)      
            
                if (not further_train_individuals):
                    for param in new_model.parameters():
                        param.requires_grad = False

                self.models += [new_model.to(device)]
                
                model_i += 1
                        
        if (self.cnn_hidden_channels2 is None):
            self.conv1 = nn.Conv1d(in_channels=len(self.models_type_rank), out_channels=self.cnn_hidden_channels1, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(in_channels = self.cnn_hidden_channels1, out_channels = 1, kernel_size=3, stride=1, padding = 1)
        else:
            self.conv1 = nn.Conv1d(in_channels=len(self.models_type_rank), out_channels=self.cnn_hidden_channels1, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(in_channels = self.cnn_hidden_channels1, out_channels = self.cnn_hidden_channels2, kernel_size=3, stride=1, padding = 1)
            self.conv3 = nn.Conv1d(in_channels = self.cnn_hidden_channels2, out_channels = 1, kernel_size=3, stride=1, padding = 1)
                                    
        
        in_dim = len(self.models_type_rank)
            
        
        if (mlp_hidden_dim2 is None):
            self.fc1 = nn.Linear(in_dim, self.mlp_hidden_dim1)
            self.fc3 = nn.Linear(self.mlp_hidden_dim1, 1)
        else:
            self.fc1 = nn.Linear(in_dim, self.mlp_hidden_dim1)
            self.fc2 = nn.Linear(self.mlp_hidden_dim1, self.mlp_hidden_dim2)
            self.fc3 = nn.Linear(self.mlp_hidden_dim2, 1)
        
        self.dropout1 = nn.Dropout(p=dropout_p_before_mlp)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.act = nn.ReLU()
    
    def set_agg_func(self, agg_func):
        self.agg_func = agg_func
        
        
    # def get_models(self): return self.models

    def forward(self, idxs):
            
        predictions = [model(idxs) for model in self.models]
                    
        predictions = torch.stack(predictions).t().float()
            
        if (self.agg_func == 'mean'):
            return predictions.mean(axis=1)
        
        elif (self.agg_func == 'median'):
            return predictions.median(axis=1).values
        
        elif (self.agg_func == 'max'):
            return predictions.max(axis=1).values
        
        elif (self.agg_func == 'min'):
            return predictions.min(axis=1).values
        
        elif (self.agg_func == 'mlp'):
                                
            predictions = self.dropout1(predictions)
            predictions = self.fc1(predictions)
            predictions = self.act(predictions)
            
            predictions = self.dropout2(predictions)
            
            if (self.mlp_hidden_dim2 is not None):
                predictions = self.fc2(predictions)
                predictions = self.act(predictions)
                predictions = self.dropout2(predictions)
            
            predictions = self.fc3(predictions)
            
            return predictions.squeeze()
        
        elif(self.agg_func == 'cnn'):
            
            predictions = predictions.t().unsqueeze(0)
            
            predictions = self.conv1(predictions)
            predictions = self.act(predictions)
            predictions = self.conv2(predictions)
            
            if (self.cnn_hidden_channels2 is not None):
                predictions = self.act(predictions)
                predictions = self.conv3(predictions)
            
            return predictions.squeeze()
        
        else:
            return predictions.median(axis=1).values
    
    def predict(self, idxs):
        return self.forward(idxs)
                    
                    
                    
def train_learned_ensemble(model = None,
                           sparse_tensor = None,
                           lr = 5e-3,
                           wd = 1e-4,
                           num_epochs = 7500,
                           batch_size = 256,
                           flags = 100,
                           early_stopping = True,
                           val_size = 0.2,
                           verbose = False,
                           epoch_display_rate = 10,
                           device = "cuda" if torch.cuda.is_available() else "cpu"):
        
        
        if (model is None):
            print("Cannot train on no model!")
            return
        
        if (sparse_tensor is None):
            print("Cannot train on no tensor!")
            return
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        train_indices = sparse_tensor.indices().t().to(device) # NNZ x mode
        train_values = sparse_tensor.values().to(device)       # NNZ
        train_values = train_values.to(torch.double)

        indices, val_indices, values, val_values = train_test_split(train_indices, train_values, test_size=val_size, random_state=18)

        dataset = COODataset(indices, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        flag = 0

        err_lst = list()
        old_MAE = 1e+6
        # train the model
        for epoch in range(num_epochs):

            model.train()
            #     epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                inputs, targets = batch[0].to(device), batch[1].to(device)
                                
                outputs = model.predict(inputs)

                loss = loss_fn(outputs.to(torch.double), targets)

                loss.backward()
                optimizer.step()
            #         epoch_loss += loss.item()

            model.eval()
            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    train_rec = model.predict(indices)
                    train_MAE = abs(train_rec - values).mean()
                    
                    val_rec = model.predict(val_indices)
                    val_MAE = abs(val_rec - val_values).mean()

                    # for early stopping
                    if (early_stopping):
                        
                        if (old_MAE < val_MAE):
                            flag +=1
                                
                        if flag == flags:
                            break
                    
                    old_MAE = val_MAE

                    
                    if (verbose and ((epoch+1)%epoch_display_rate==0)): 
                        print(f"Epoch {epoch+1} Train_MAE: {train_MAE:.4f} Val_MAE: {val_MAE:.4f}\t")
                        
        return model