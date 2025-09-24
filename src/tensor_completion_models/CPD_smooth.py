import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn import functional as F
import numpy as np

from src.tensor_completion_models.codes_costco_tucker.costco import *
from src.tensor_completion_models.codes_costco_tucker.read import *
from src.tensor_completion_models.codes_costco_tucker.utils import *

from sklearn.utils.validation import check_random_state

import tensorly as tl

tl.set_backend('pytorch')
random_state = 1234
rng = check_random_state(random_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_random(size):
    ''' Make random values with a given size'''

    #if len(size) >1, size must be tuple
    random_value = torch.FloatTensor(rng.random_sample(size))
    return random_value

class Kernel(nn.Module):
    ''' Implement a kernel smoothing regularization'''
    
    def __init__(self, window, density, sigma = 0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(device)
        
    def gaussian(self):
        ''' Make a Gaussian kernel'''

        window = int(self.window-1)/2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window+1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1).to(torch.double)

    
    def forward(self,factor):
        ''' Perform a Gaussian kernel smoothing on a temporal factor'''

        row, col = factor.shape
        conv = F.conv2d(factor.view(1, 1, row, col), self.weight, 
                          padding = (int((self.window-1)/2), 0))
        return conv.view(row, col)
    
    
class Inverse_Kernel(nn.Module):
    ''' Implement a kernel smoothing regularization'''
    
    def __init__(self, window, density, sigma = 0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(device)
        
    def gaussian(self):
        ''' Make a Gaussian kernel'''

        window = int(self.window-1)/2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window+1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1).to(torch.double)

    
    def forward(self,factor):

        row, col = factor.shape
        
        factor = factor.T
        conv = F.conv2d(factor.view(1, 1, col, row), self.weight, 
                          padding = (int((self.window-1)/2), 0))
        

        return conv.view(row, col)


class CPD_Smooth(nn.Module):

    def __init__(self, cfg):
        super(CPD_Smooth, self).__init__()

        self.cfg = cfg
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.nmode = len(self.sizes)
        
        self.window = cfg.window
        self.inverse_window = cfg.inverse_window

        # Factor matrices
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])

        # self._initialize()
        self.smooth = Kernel(self.window, density = None).to(device)
        self.inverse_smooth = Inverse_Kernel(self.inverse_window, density = None).to(device)


    def _initialize(self):
        rng = check_random_state(self.cfg.random)
        for m in range(self.nmode):
            self.embeds[m].weight.data = torch.tensor(rng.random_sample((self.sizes[m], self.rank)))


    def recon(self, idxs):
        '''
        Reconstruct a tensor entry with a given index
        '''
        # Element-wise product and sum
        facs = [self.embeds[m](idxs[:, m]).unsqueeze(-1) for m in range(self.nmode)]
        concat = torch.concat(facs, dim=-1) # NNZ x rank x nmode
        rec = torch.prod(concat, dim=-1)    # NNZ x ranak
        return rec.sum(-1)

    def forward(self, idxs):
        return self.recon(idxs)
    
    
    def inverse_std_error(self, mode):
        return self.embeds[mode].weight.std(axis = 1).sum()
    
    # from TATD
    
    def smooth_reg(self, mode):
        ''' Perform a smoothing regularization on the time factor '''
        
        smoothed = self.smooth(self.embeds[mode].weight)

        sloss = (smoothed - self.embeds[mode].weight).pow(2)
                
        # if self.sparse == 1:
        #    sloss = sloss * self.density.view(-1, 1) 
        
        return sloss.sum()
    
    def inverse_smooth_reg(self, mode):
        ''' Perform a smoothing regularization on the time factor '''
        
        smoothed = self.inverse_smooth(self.embeds[mode].weight)

        sloss = (smoothed - self.embeds[mode].weight).pow(2)
                
        # if self.sparse == 1:
        #    sloss = sloss * self.density.view(-1, 1) 
        
        return sloss.sum()