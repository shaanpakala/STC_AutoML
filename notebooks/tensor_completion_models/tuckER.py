# work_dir = "/home/spaka002/NSF_REU_2024/"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from notebooks.tensor_completion_models.codes_costco_tucker.read import *
from notebooks.tensor_completion_models.codes_costco_tucker.utils import *

from sklearn.utils.validation import check_random_state
import tensorly as tl



class TuckER(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super(TuckER, self).__init__()
        self.config = config
        self.rank = config.rank
        self.sizes = config.sizes
        self.embeds = nn.ModuleList([nn.Embedding(s, self.rank) for s in self.sizes])
        if len(self.sizes) == 4:
            self.W = torch.nn.Parameter(torch.randn([self.rank, self.rank, self.rank, self.rank]))
        else:
            self.W = torch.nn.Parameter(torch.randn([self.rank, self.rank, self.rank]))

        self.input_dropout = torch.nn.Dropout(config["in_drop"])
        self.hidden_dropout1 = torch.nn.Dropout(config["hidden_drop"])
        self.hidden_dropout2 = torch.nn.Dropout(config["hidden_drop"])
        if len(self.sizes) == 4:
            self.hidden_dropout3 = torch.nn.Dropout(config["hidden_drop"])
  
        self.bn0 = torch.nn.BatchNorm1d(self.rank)
        self.bn1 = torch.nn.BatchNorm1d(self.rank)
        if len(self.sizes) == 4:
            self.bn2 = torch.nn.BatchNorm1d(self.rank)

    def forward(self, idxs):

        if len(self.sizes) != 4:
            e1 = self.embeds[0](idxs[:, 0])
            x = self.bn0(e1)
            x = self.input_dropout(x)
            x = x.view(-1, 1, self.rank)

            r = self.embeds[1](idxs[:, 1])
            W_mat = torch.mm(r, self.W.view(r.size(1), -1))
            W_mat = W_mat.view(-1, self.rank, self.rank)
            W_mat = self.hidden_dropout1(W_mat)

            x = torch.bmm(x, W_mat) 
            x = x.view(-1, self.rank)      
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            # print(x.view(-1, 1, self.rank).shape, self.embeds[-1](idxs[:, 2]).view(-1, self.rank, 1).shape )
            x = torch.bmm(x.view(-1, 1, self.rank), self.embeds[-1](idxs[:, 2]).view(-1, self.rank, 1))
            pred = torch.sigmoid(x.view(-1))
            return pred
        else:

            # Prepare embeds
            e1 = self.embeds[0](idxs[:, 0])
            e1 = self.bn0(e1)
            e1 = self.input_dropout(e1)
            e1 = e1.view(-1, 1, self.rank) # NNZ x 1 x Rank

            e2 = self.embeds[1](idxs[:, 1])
            e2 = e2.view(-1, 1, self.rank) # NNZ x 1 x Rank

            e3 = self.embeds[2](idxs[:, 2])
            e3 = e3.view(-1, 1, self.rank) # NNZ x 1 x Rank

            # Start operations
            r = self.embeds[-1](idxs[:, -1])
            W_mat = torch.mm(r, self.W.view(r.size(1), -1))
            W_mat = self.hidden_dropout1(W_mat)

            x1 = torch.bmm(e1, W_mat.view(-1, self.rank, self.rank * self.rank))
            x1 = self.bn1(x1.reshape(-1, self.rank, self.rank))
            x1 = self.hidden_dropout2(x1)
            # print(x1.shape)

            x2 = torch.bmm(e2, x1)
            x2 = self.bn2(x2.reshape(-1, self.rank, 1))
            x2 = self.hidden_dropout2(x2)
            # print(x2.shape)

            x3 = torch.bmm(e3, x2)
            pred = torch.sigmoid(x3.view(-1))

        return pred

def eval_data(dataloader, model):
    rec = []
    true = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0], batch[1]
            outputs = model(inputs)
            rec.append(outputs)
            true.append(targets)
    return torch.hstack(rec), torch.hstack(true)

def train_tuckER(tensor_indices, tensor_values, cfg, flags = 15, wandb=None, verbose=False, epoch_display_rate = 1, val_size = 0.2, for_queries = False):

    if (verbose): print("Training TuckER....")
    
    train_i, valid_i, train_v, valid_v = train_test_split(tensor_indices, tensor_values, test_size = val_size, shuffle=True, random_state = 18)

    # Batch loader
    train_dataset = COODataset(train_i, train_v)
    valid_dataset = COODataset(valid_i, valid_v)
    # test_dataset = COODataset(test_i, test_v)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.bs, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=True)

    # Set hyperparameters
    lr = cfg.lr
    wd = cfg.wd
    epochs = cfg.epochs

    model = TuckER(cfg).to(cfg.device)
    
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    total_running_time = 0
    # start = time.time()
    flag = 0
    flag_2 = 0
    old_valid_MAE = 1e+6
    for epoch in range(epochs):
        # epoch_start = time.time()
        epoch_loss = 0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0], batch[-1]
            outputs = model(inputs)
            
            if (not for_queries):
                outputs = outputs.to(float)
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # epoch_end = time.time()
        # total_running_time += (epoch_end - epoch_start)
        model.eval()    
        if (epoch+1) % epoch_display_rate == 0:
            with torch.no_grad():
                train_rec, train_true = eval_data(train_dataloader, model)
                val_rec, val_true = eval_data(valid_dataloader, model)
                train_MAE = abs(train_rec - train_true).mean()
                valid_MAE = abs(val_rec - val_true).mean()

                if verbose:
                   print(f"Epochs {epoch+1} Train_MAE: {train_MAE:.4f}\t"
                          f"Valid_MAE: {valid_MAE:.4f}\t")

                # if wandb:
                #     wandb.log({"train_rmse":train_rmse,
                #             "valid_rmse":valid_rmse,
                #             "total_running_time":total_running_time})

                if (old_valid_MAE < valid_MAE):
                    flag +=1
                    
                if (abs((old_valid_MAE - valid_MAE)/old_valid_MAE) < 1e-4):
                    flag_2 += 1
                    
                if flag == flags:
                    break
                
                if flag_2 == 25:
                    break

                old_valid_MAE = valid_MAE
                
    
    return model