import os
import numpy as np
from pathlib import Path
from dotmap import DotMap
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class COODataset(Dataset):
    def __init__(self, idxs, vals):
        self.idxs = idxs
        self.vals = vals

    def __len__(self):
        return self.vals.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.vals[idx]


def read_data(cfg, bin_val=True, neg=True, sparsify=False):
    ''' Read tensors in COO format.
        cfg: configuration file
        bin_val: binary value 
        neg: include negatvie sampling
    '''

    dct = DotMap()
    dct.bin_val = bin_val
    name = cfg.dataset
    device = cfg.device
    data_path = os.path.join(Path.home(), "NSF_REU_2024/Research", name)

    if name.startswith('epigenom'):
        bin_val = False
        neg = False

    for dtype in ['train', 'valid', 'test']:
        if cfg.verbose:
            print(f"Reading {dtype} dataset----------------------")

        idxs_lst = []
        vals_lst = []

        idxs = np.load(f'{data_path}/{dtype}_idxs.npy')
        if bin_val:
            vals = torch.ones(idxs.shape[0])
        else:
            vals = np.load(f'{data_path}/{dtype}_vals.npy')

        idxs_lst.append(idxs)
        vals_lst.append(vals)
        
        if neg:
            if cfg.verbose:
                print(f"Read negative samples")
            neg_path = os.path.join(data_path, 'neg_sample0')
            neg_idxs = np.load(f'{neg_path}/{dtype}_idxs.npy')
            neg_vals = np.zeros(neg_idxs.shape[0])
            idxs_lst.append(neg_idxs)
            vals_lst.append(neg_vals)

    
        total_idxs = np.vstack(idxs_lst)
        total_vals = np.hstack(vals_lst)

        dct[f'{dtype}_i'] = torch.LongTensor(total_idxs).to(device)
        dct[f'{dtype}_v'] = torch.FloatTensor(total_vals).to(device)   
            
    dct.sizes = get_size(name)
    
    if sparsify:
        train_data = torch.hstack([dct.train_i.cpu(), dct.train_v.cpu().reshape(-1, 1)]).numpy()
        train, test = train_test_split(train_data, train_size=cfg.sparsity, random_state=1)
        dct['train_i']  = torch.LongTensor(train[:, :-1]).to(device)
        dct['train_v']  = torch.FloatTensor(train[:, -1]).to(device)
    print(f"Dataset: {cfg.dataset} "
          f"|| size: {dct.sizes} & training observed x (pos + neg): {dct[f'train_v'].shape[0]}")    

    return dct
    

def get_size(name):
    '''
    Get size (dimensionality) of tensor.
    name: dataset name
    '''
    
    if name == "ml": 
        size = [610, 9724, 4110]
        
    if name == "yelp": 
        size = [70818, 15580, 109]
        
    if name == "foursquare_nyc":
        size = [1084, 38334, 7641]
        
    if name == "foursquare_tky":
        size = [2294, 61859, 7641]
        
    if name == "yahoo_msg":
        size = [82309, 82308, 168]

    if name == "epigenom":
        size = [5, 5, 1000]

    if name.endswith('dblp3'):
        size = [4057, 14328, 7723]

    if name.endswith('dblp4'):
        size = [4057, 14328, 7723, 20]

    if name == 'dblp':
        size = [3514, 11192, 2931, 18]


    if name == 'trans_dblp':
        size = [3514, 11192, 2931, 18]


    if name == 'dblp2':
        size = [3514, 11192, 18]

    if name == 'trans_dblp2':
        size = [3514, 11192, 18]

    return size
