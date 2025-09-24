
import time
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.tensor_completion_models.codes_costco_tucker.utils import *
from src.tensor_completion_models.codes_costco_tucker.read import *

class CoSTCo(nn.Module):
    def __init__(self, cfg):
        super(CoSTCo, self).__init__()
        nc = cfg.nc
        self.loss = cfg.loss
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])
        self.conv1 = nn.Conv2d(1, nc, kernel_size=(1, len(self.sizes)), padding=0)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=(self.rank, 1), padding=0)
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self._initialize()
        
    def _initialize(self):
        
        for i in range(len(self.embeds)):
            nn.init.kaiming_uniform_(self.embeds[i].weight.data)
        
    def forward(self, inputs):
        '''
        inputs: indices of nnz
        '''
        
        embeds = [self.embeds[m](inputs[:, m]).reshape(-1, self.rank, 1)
                  for m in range(len(self.sizes))]
        x = torch.cat(embeds, dim=2)
        x = x.reshape(-1, 1, self.rank, len(self.sizes))# NNZ_batch x 1 x rank x nmode 
        
        # CNN on stacked embeds
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # Aggregate with mlps
        x = x.view(-1, x.size(1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.loss == 'bceloss':
            x = self.sigmoid(x)
        else:
            x = self.relu(x)
        return x.reshape(-1)
    


def train(tensor, cfg, wandb=None, verbose=False):

    print("Training CoSTCo....")

    train_i, train_v = tensor.train_i, tensor.train_v
    valid_i, valid_v = tensor.valid_i, tensor.valid_v
    test_i,  test_v  = tensor.test_i,  tensor.test_v
    shape = tensor.sizes

    # Batch loader
    dataset = COODataset(train_i, train_v)
    dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=True)

    # Set hyperparameters
    lr = cfg.lr
    wd = cfg.wd
    epochs = cfg.epochs

    model = CoSTCo(cfg).to(cfg.device)
    if cfg.loss == 'bceloss':
        loss_fn = nn.BCELoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    flag = 0
    old_valid_rmse = 1e+6
    total_running_time = 0
    start  = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0], batch[-1]
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_end = time.time()
        total_running_time += (epoch_end - epoch_start)

        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                val_rec = model(valid_i)
                train_rmse = np.sqrt(epoch_loss/len(dataloader))
                valid_rmse = rmse(val_rec, valid_v)

                if verbose:
                   print(f"Epochs {epoch} TrainRMSE: {train_rmse:.4f}\t"
                          f"ValidRMSE: {valid_rmse:.4f}\t")

                if wandb:
                    wandb.log({"train_rmse":train_rmse,
                            "valid_rmse":valid_rmse,
                            "total_running_time":total_running_time})

                if (old_valid_rmse < valid_rmse):
                    flag +=1
                    
                if flag == 10:
                    break
                
                old_valid_rmse = valid_rmse
    training_time = time.time() - start
    model.eval()
    with torch.no_grad(): 
        test_rec =  model(test_i)
        if tensor.bin_val:
            result = eval_(test_rec.data, test_v)
            result['test_rmse'] = rmse(test_rec, test_v)
        else:
            result={'test_rmse':rmse(test_rec, test_v)}
        if wandb:
            result['avg_running_time'] = total_running_time / epoch
            result['all_total_training_time'] = training_time
            wandb.log(result)

    model.result = result

    return model
