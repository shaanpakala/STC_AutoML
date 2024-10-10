import os
import sys

sys.path.append(os.getcwd().split('notebooks')[0])

from notebooks.utilities.utils import *
from notebooks.utilities.helper_functions import *

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from notebooks.tensor_completion_models.codes_costco_tucker.read import *

import tensorly as tl
from tensorly.decomposition import TensorTrain

tl.set_backend('pytorch')



class tensor_train_completion(nn.Module):
    def __init__(self, rank, tensor_size):
        super(tensor_train_completion, self).__init__()

        self.rank = rank
        self.sizes = tensor_size
        self.nmode = len(self.sizes)
        
        TT_decom = TensorTrain(rank = rank)
        decomp = TT_decom.fit_transform(torch.rand(self.sizes))
        factors = decomp.factors
        self.factors  = nn.ParameterList([nn.Parameter(torch.rand(factor.shape)) for factor in factors])
        
        del TT_decom, decomp, factors

    def full_recon(self):
        
        # Reconstruct a tensor
        full_tensor = tl.tt_to_tensor(self.factors)

        # # normalize tensor reconstruction
        # full_tensor = (full_tensor - full_tensor.min())/(full_tensor.max() - full_tensor.min())
        
        return full_tensor
    
    def recon(self, idxs):
        full_tensor = self.full_recon()
        return torch.stack([full_tensor[tuple(idx)] for idx in idxs])
    
    def forward(self, idxs):
        return self.recon(idxs)     
    
    

def train_tensor_train(sparse_tensor, 
                       rank = 3,
                       lr = 1e-3, 
                       wd = 1e-4, 
                       num_epochs = 5000, 
                       batch_size = 256, 
                       early_stopping = True, 
                       flags = 25, 
                       val_size = 0.2, 
                       verbose = False, 
                       epoch_display_rate = 1,
                       device = 'cpu'):
    
    
    model = tensor_train_completion(rank = rank, 
                                    tensor_size = sparse_tensor.size())
    
    train_indices = sparse_tensor.indices().t()
    train_values = sparse_tensor.values()
    
    mask = torch.zeros(sparse_tensor.size(), dtype=torch.int32)
    mask[tuple(sparse_tensor.indices())] = 1

    # TODO
    # train_sparse_tensor_dense = model.sparse_tensor.to_dense()
    # train_mask_tensor = (sparse_tensor_dense != 0).int()
    # valid, test as well
        
    training_indices = train_indices.to(device) # NNZ x mode
    training_values = train_values.to(device)   # NNZ
    training_values = training_values.to(torch.double)

    indices, val_indices, values, val_values = train_test_split(training_indices, training_values, test_size=val_size, random_state=18)

    dataset = COODataset(indices, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    flag = 0
    old_val_MAE = 1e6

    for epoch in range(num_epochs):

        model.train()
        for batch in dataloader:
            
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            rec = model(inputs)
                    
            loss = (((rec - targets))**2).sum()
                    
            loss.backward()
            optimizer.step()
        
        val_MAE = abs(model(val_indices) - val_values).mean()
            
        if (verbose and (epoch+1)%epoch_display_rate == 0):
            print(f"Epoch {epoch+1} Loss {loss:.4f} Val_MAE {val_MAE:.4f}")
            
            
        if (early_stopping):
            
            if (val_MAE > old_val_MAE):
                flag += 1
            
            if (flag >= flags): break
        
        old_val_MAE = val_MAE + 0
    
    return model