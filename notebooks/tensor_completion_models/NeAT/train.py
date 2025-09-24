

import pdb
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from notebooks.tensor_completion_models.NeAT.read import *
from notebooks.tensor_completion_models.NeAT.model import *
from notebooks.tensor_completion_models.NeAT.utils import *
from notebooks.tensor_completion_models.NeAT.metrics import *


def train_NeAT(train_indices, train_values, cfg, verbose=True):

    if (verbose): print("Start training a Neural additive Tensor Decomposition (NeAT) ...!")
    # Prepare dataset
    training_indices = train_indices.to(cfg.device) # NNZ x mode
    training_values = train_values.to(cfg.device)       # NNZ
    # training_values = training_values.to(torch.double)
    
    indices, val_indices, values, val_values = train_test_split(training_indices, training_values, test_size=cfg.val_size, random_state=18)

    dataset = COODataset(indices, values)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # create the model
    # m = nn.Sigmoid()
    # if cfg.lossf == 'BCELoss':
        # if (verbose): print("loss_fn: BCE")
        # loss_fn = nn.BCELoss(reduction='mean')
    # else:
        # if (verbose): print("loss_fn: MSE")
    loss_fn = nn.MSELoss()

    model = NeAT(cfg, cfg.sizes).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    flag=0
    old_valid_MAE = 1e+6

    for epoch in range(cfg.epochs):

        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0], batch[1]
            outputs = model(inputs)
            loss = loss_fn(outputs.to(torch.double), targets.to(torch.double))
             
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                val_rec = model(val_indices)
                valid_MAE = abs(val_rec - val_values).mean()

                if verbose:
                    print(f"Epoch {epoch+1} Val MAE: {valid_MAE:.4f}")

                if (old_valid_MAE <= valid_MAE):
                    flag +=1
                    
                old_valid_MAE = valid_MAE
                
                if flag == 5:
                    break

    return model
