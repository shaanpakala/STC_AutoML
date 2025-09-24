import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from notebooks.tensor_completion_models.codes_costco_tucker.costco import *
from notebooks.tensor_completion_models.codes_costco_tucker.read import *
from notebooks.tensor_completion_models.codes_costco_tucker.utils import *

from sklearn.utils.validation import check_random_state
import tensorly as tl

class CoSTCo(nn.Module):
    
    def __init__(self, cfg):
        super(CoSTCo, self).__init__()
        nc = cfg.nc
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])
        self.conv1 = nn.Conv2d(1, nc, kernel_size=(1, len(self.sizes)), padding=0)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=(self.rank, 1), padding=0)
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.relu = nn.ReLU()
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
        x = self.relu(x)
        
        return x.reshape(-1)