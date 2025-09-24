import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from notebooks.tensor_completion_models.codes_costco_tucker.costco import *
from notebooks.tensor_completion_models.codes_costco_tucker.read import *
from notebooks.tensor_completion_models.codes_costco_tucker.utils import *

from sklearn.utils.validation import check_random_state
import tensorly as tl

class CPD(nn.Module):

    def __init__(self, cfg):
        super(CPD, self).__init__()

        self.cfg = cfg
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.nmode = len(self.sizes)

        # Factor matrices
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])
        # self.initialize_()

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