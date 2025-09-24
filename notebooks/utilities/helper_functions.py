# work_dir = "/home/spaka002/NSF_REU_2024/"

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score

import os
import sys
sys.path.append(os.getcwd().split('notebooks')[0])

from notebooks.tensor_completion_models.ETC import *
from notebooks.tensor_completion_models.CoSTCo import *
from notebooks.tensor_completion_models.CPD import *
from notebooks.tensor_completion_models.CPD_smooth import *
from notebooks.tensor_completion_models.NeAT.train import *

from sklearn.utils.validation import check_random_state
import tensorly as tl

from sklearn.model_selection import train_test_split, KFold



def train_tensor_completion(model_type, 
                            sparse_tensor,
                            rank = 5, 
                            num_epochs = 15_000, 
                            batch_size = 256, 
                            lr = 5e-3, 
                            wd = 5e-4, 
                            loss_p = 2,
                            zero_lambda = 1,
                            cpd_smooth_lambda = 2,
                            cpd_smooth_window = 3,
                            cpd_inverse_smooth_lambda = 0,
                            cpd_inverse_smooth_window = 3,
                            cpd_inverse_std_lambda = 0,
                            non_smooth_modes = list(),
                            non_inverse_smooth_modes = list(),
                            NeAT_hidden_dim = 32,
                            NeAT_drop = 0.1,
                            NeAT_drop2 = 0.5,
                            tucker_in_drop = 0.1,
                            tucker_hidden_drop = 0.1,
                            train_norm = None,
                            early_stopping = True,
                            flags = 15,
                            verbose = False, 
                            epoch_display_rate = 1, 
                            val_size = 0.2,
                            return_errors = False,
                            reinitialize_count = 0,
                            convert_to_cpd = False,
                            for_queries = False,
                            device = "cuda" if torch.cuda.is_available() else "cpu"):


    if (verbose): print(f"Rank = {rank}; lr = {lr}; wd = {wd}\n")

    train_indices = sparse_tensor.indices().t()
    train_values = sparse_tensor.values()
    tensor_size = sparse_tensor.size()

    training_indices = train_indices.to(device) # NNZ x mode
    training_values = train_values.to(device)   # NNZ
    training_values = training_values.to(torch.double)
    
    indices, val_indices, values, val_values = train_test_split(training_indices, training_values, test_size=val_size, random_state=18)

    cfg = DotMap()

    cfg.norm = lambda x: x
    cfg.unnorm = lambda x: x

    if train_norm is not None:

        if train_norm == 'minmax':

            max_ = training_values.max()
            min_ = training_values.min()

            cfg.norm = lambda x: (x - min_)/(max_ - min_)
            cfg.unnorm = lambda x: (x*(max_ - min_)) + min_

        elif train_norm == 'standard':

            mean_ = training_values.mean()
            std_ = training_values.std()

            cfg.norm = lambda x: (x - mean_)/(std_)
            cfg.unnorm = lambda x: (x*std_) + mean_


    values = cfg.norm(values)
    val_values = cfg.norm(val_values)
    
    dataset = COODataset(indices, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    cfg.nc = training_indices.shape[0]
    cfg.rank = rank
    cfg.sizes = tensor_size
    cfg.lr = lr
    cfg.wd = wd
    cfg.epochs = num_epochs
    cfg.random = 18

    # create the model
    
    if (model_type == 'cpd'):
        model = CPD(cfg).to(device)
        
    elif (model_type == 'cpd.smooth'):
        
        cfg.smooth_lambda = cpd_smooth_lambda
        cfg.window = cpd_smooth_window
        cfg.inverse_window = cpd_inverse_smooth_window
        cfg.inverse_smooth_lambda = cpd_inverse_smooth_lambda
        cfg.inverse_std_lambda = cpd_inverse_std_lambda
        
        model = CPD_Smooth(cfg).to(device)
            
        for param in model.parameters():
            param.requires_grad = True

    elif (model_type == 'costco'):
        model = CoSTCo(cfg).to(device)
        
    
    elif (model_type == 'NeAT'):
        
        cfg.rank = rank
        cfg.sizes = tensor_size
        cfg.layer_dims = [len(cfg.sizes), NeAT_hidden_dim, 1]
        cfg.depth = len(cfg.layer_dims)
        cfg.lr = lr
        cfg.wd = wd
        cfg.dropout = NeAT_drop
        cfg.dropout2 = NeAT_drop2
        cfg.epochs = num_epochs
        cfg.batch_size = batch_size
        cfg.device = device
        cfg.val_size = val_size

        model = NeAT(cfg, sizes = cfg.sizes).to(device)

    else:
        print("No Model Selected!")
        model = CPD(cfg).to(device)

    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    model._initialize()

    flag = 0
    flag_2 = 0

    err_list = list()
    old_MAE = 1e+6

    # train the model
    for epoch in range(cfg.epochs):

        model.train()

        for batch in dataloader:

            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            model = model.to(device)

            rec = model(inputs)

# _________________________ loss_fn ____________________________________________________________________________________________

            if model_type == 'tuckER':

                loss = bce_loss(rec, targets.to(torch.float))

            else:

                errors = abs(rec - targets).pow(loss_p)
                
                zero_lambda_mask = ((targets == 0) * zero_lambda) + (targets != 0)
                errors = errors * zero_lambda_mask
                
                loss = errors.sum()
                
                if (model_type == 'cpd.smooth' or model_type == 'cpd.smooth.t'):
                    
                    for n in range(len(tensor_size)):    
                        if n not in non_smooth_modes:                
                            loss = loss + (model.smooth_reg(n) * cfg.smooth_lambda)
                            
                        if n not in non_inverse_smooth_modes:
                            loss = loss + (model.inverse_smooth_reg(n) * cfg.inverse_smooth_lambda)
                            loss = loss + (model.inverse_std_error(n) * cfg.inverse_std_lambda)

# ______________________________________________________________________________________________________________________________


            loss.backward()
            optimizer.step()

        model.eval()
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                train_rec = model(indices)
                train_MAE = abs(train_rec - values).mean()
                
                val_rec = model(val_indices)
                val_MAE = abs(val_rec - val_values).mean()

                # for early stopping
                if (early_stopping):
                    
                    if (old_MAE < val_MAE):
                        flag +=1
                            
                    if flag == flags:
                        break
                    
                    if (old_MAE == val_MAE):
                        flag_2 +=1
                        
                if flag_2 == 25:
                    break
                
                old_MAE = val_MAE

                    
                err_list += [old_MAE]
                
                if (verbose and ((epoch+1)%epoch_display_rate==0)): 
                    print(f"Epoch {epoch+1} Train_MAE: {train_MAE:.4f} Val_MAE: {val_MAE:.4f}\t")
                
    if (verbose): print()                       
    
    
    # reinitialize model if it didn't converge!
    if (torch.tensor(err_list[10:]).std() < 1e-6): 
        
        if (reinitialize_count >= 5) and convert_to_cpd:
            
            if (verbose): print(f"\nConverting {model_type} to cpd!\n")

            return train_tensor_completion(model_type = 'cpd', 
                                            sparse_tensor = sparse_tensor,
                                            rank = rank, 
                                            num_epochs = num_epochs, 
                                            batch_size = batch_size, 
                                            lr=lr, 
                                            wd=wd, 
                                            tucker_in_drop = tucker_in_drop,
                                            tucker_hidden_drop = tucker_hidden_drop,
                                            early_stopping = early_stopping, 
                                            flags = flags, 
                                            verbose = verbose, 
                                            epoch_display_rate = epoch_display_rate, 
                                            val_size = val_size,
                                            return_errors = return_errors,
                                            reinitialize_count=reinitialize_count+1)
        
        if (verbose): print(f"\nReinitializing {model_type}! Reinitialize Count: {reinitialize_count}.\n")     

        return train_tensor_completion(model_type = model_type, 
                            sparse_tensor = sparse_tensor,
                            rank = rank, 
                            num_epochs = num_epochs, 
                            batch_size = batch_size, 
                            lr=lr, 
                            wd=wd, 
                            tucker_in_drop = tucker_in_drop,
                            tucker_hidden_drop = tucker_hidden_drop,
                            early_stopping = early_stopping, 
                            flags = flags, 
                            verbose = verbose, 
                            epoch_display_rate = epoch_display_rate, 
                            val_size = val_size,
                            return_errors = return_errors,
                            reinitialize_count=reinitialize_count+1)
                            
    if (return_errors): return model, torch.tensor(err_list)
    return model


# device = "cuda" if torch.cuda.is_available() else "cpu"


def get_subset(x, y, portion=1, random_state=18):
    
        if (portion>=1): return x, y
        
        X, tx, Y, ty = train_test_split(x, y, test_size=(1-portion), random_state=random_state)
        del tx, ty
        
        return X, Y

def return_eval(model, x, y, metric = 'f1', n_splits = 5, change_input = None, smote_train=False, random_state = 18):
    
    if change_input is not None:
        mapping = change_input
        mapping_tensor = torch.tensor([mapping[i] for i in range(1, max(mapping.keys()) + 1)])
        x = mapping_tensor[x - 1]
    
    kf = KFold(n_splits=n_splits, shuffle = True, random_state = random_state)
        
    overall_metric = 0
    for i, (train_index, test_index) in enumerate(kf.split(x)):

        X_train = x[train_index]
        X_test = x[test_index]
        
        Y_train = y[train_index]
        Y_test = y[test_index]
                
        if (smote_train):
            smote = SMOTE(random_state=random_state)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)

        
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        
        # _______ insert evaluation metric ___________________________________
        
        if metric == 'f1':
            
            if (len(set(y)) == 2): average = 'binary'
            else: average = 'weighted'
                            
            metric_value = f1_score(Y_test, preds, average = average)
            
        elif metric == 'accuracy':
            metric_value = accuracy_score(Y_test, preds)
            
        elif metric == 'precision':
            metric_value = precision_score(Y_test, preds)
            
        elif metric == 'recall':
            metric_value = recall_score(Y_test, preds)
            
        elif metric == 'auroc':
            pred_prob = model.predict_proba(X_test)
            metric_value = roc_auc_score(Y_test, pred_prob[:, 1])
                        
        elif metric == 'mae':
            metric_value = abs(Y_test - preds).mean()
            
        elif metric == 'mse':
            metric_value = ((Y_test - preds)**2).mean()
            
        elif metric == 'r2':
            metric_value = r2_score(Y_test, preds)
            
        else:
            
            print("No metric value for return_eval() !")
            return 0
        
        # ____________________________________________________________________
        
        overall_metric += metric_value
            
    return (overall_metric/n_splits)


rand_index = lambda x: tuple([int(random.random()*i) for i in x])

def get_rand_indices(shape, num_indices):

    index_bank = dict()

    while (len(index_bank) < num_indices):
        index_bank[rand_index(shape)] = True

    return list(index_bank)


# completely randomly generate sparse tensor, such that there are 'portion' non zero values remaining

def get_sparse_tensor(t, portion = 0.05, verbose = False, device = "cuda" if torch.cuda.is_available() else "cpu"):

    total_cells = 1
    for s in t.shape: total_cells*=s

    portion_of_entries = portion
    indices = get_rand_indices(t.shape, int(total_cells*(1-portion_of_entries)))
    for index in indices: t[index] = -1

    sparse_indices = (t != -1).to(int).to_sparse().indices()
    sparse_values = list()
    for index in sparse_indices.t(): sparse_values += [t[tuple(index)]]
    sparse_values = torch.tensor(sparse_values)

    sparse_tensor = torch.sparse_coo_tensor(indices = sparse_indices, values= sparse_values, size = t.size()).coalesce()
    
    return sparse_tensor.to(device)



# completely randomly generate sparse tensor, such that there are 'portion' non zero values remaining

def get_sparse_filling_in_values(t, 
                                 portion = 0.05, 
                                 filling_in_values = 'random',
                                 verbose = False, 
                                 device = "cuda" if torch.cuda.is_available() else "cpu"):

    total_cells = 1
    for s in t.shape: total_cells*=s

    portion_of_entries = portion
    indices = get_rand_indices(t.shape, int(total_cells*(1-portion_of_entries)))
    
    if (filling_in_values == 'random'):
        for index in indices: t[index] = random.random()
        
    elif (filling_in_values == 'avg_random'):
        temp_t = t.clone()
        
        for index in indices: temp_t[index] = 0
        
        sparse_t = temp_t.to_sparse()
        avg = sparse_t.values().mean()
        std = sparse_t.values().std()
        
        del sparse_t, temp_t
        
        for index in indices: t[index] = max( ( avg + ((random.random() - 0.5) * std) ) ,  0)
    
    else:
        for index in indices: t[index] = 0.65

    return t.to(device)


def get_simulated_sparse_tensor(t, portions, verbose = False, device = "cuda" if torch.cuda.is_available() else "cpu"):

    new_rand_index = lambda x, first_axis_value: tuple([first_axis_value]+[int(random.random()*i) for i in x[1:]])

    def get_indices(shape, num_indices, first_axis_value):

        index_bank = dict()

        while (len(index_bank) < num_indices):
            index_bank[new_rand_index(shape, first_axis_value=first_axis_value)] = True

        return list(index_bank)

    total_cells = 1
    for i in list(t.shape): total_cells *= i

    indices = list()

    num_values_in_axis = int(total_cells/(t.shape[0]))
    for i in range(len(portions)):

        current_portion = portions[i]
        num_indices = int(num_values_in_axis * (1-current_portion))
        if (verbose): print(f"Axis: 0; Value: {i} --> {num_indices}/{num_values_in_axis} = {float(num_indices/num_values_in_axis):.3f} empty cells.")
        indices += get_indices(t.shape, num_indices, first_axis_value = i)

    for index in indices: t[index] = -1

    sparse_indices = (t != -1).to(int).to_sparse().indices()
    sparse_values = list()
    for index in sparse_indices.t(): sparse_values += [t[tuple(index)]]
    sparse_values = torch.tensor(sparse_values)

    sparse_tensor = torch.sparse_coo_tensor(indices = sparse_indices, values= sparse_values, size = t.size()).coalesce()
    
    if (verbose): print(f"\n{len(sparse_tensor.indices().t())}/{int(total_cells)} = {len(sparse_tensor.indices().t())/total_cells:.4f} non-zero values.")

    return sparse_tensor.to(device)


def warm_start_biased_sampling(t, 
                               empty_slice = 0, 
                               empty_slice_random_portion = 0.01, 
                               other_portions = 0.15, 
                               completely_random = True,
                               verbose = False,
                               device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    portions = [other_portions for x in range(t.shape[0])]
    portions[empty_slice] = 0
    
    new_rand_index = lambda x, first_axis_value: tuple([first_axis_value]+[int(random.random()*i) for i in x[1:]])

    # function to get {num_indices} random indices, such that they all have the first axis value = {first_axis_value}
    def get_indices(shape, num_indices, first_axis_value):

        index_bank = dict()

        while (len(index_bank) < num_indices):
            index_bank[new_rand_index(shape, first_axis_value=first_axis_value)] = True

        return list(index_bank)

    total_cells = 1
    for i in list(t.shape): total_cells *= i


    # randomly remove indices according to given proportions
    indices = list()
    num_values_in_axis = int(total_cells/(t.shape[0]))
    for i in range(len(portions)):

        current_portion = portions[i]
        num_indices = int(num_values_in_axis * (1-current_portion))
        if (verbose): print(f"Axis: 0; Value: {i} --> {num_indices}/{num_values_in_axis} = {float(num_indices/num_values_in_axis):.3f} empty cells.")
        indices += get_indices(t.shape, num_indices, first_axis_value = i)

    for index in indices: t[index] = -1
    
    
    # now generate random values for the empty slice
    num_indices = int(num_values_in_axis * empty_slice_random_portion)
    if (verbose): print(f"\nAxis: 0; Value: {empty_slice} --> {num_indices}/{num_values_in_axis} = {float(num_indices/num_values_in_axis):.3f} random cells.")
    
    indices = list()
    indices = get_indices(t.shape, num_indices, first_axis_value=empty_slice)
    
    if (completely_random):
        for index in indices: t[index] = random.random()
        
    else:
        # convert values to sparse tensor format
        sparse_indices = (t != -1).to(int).to_sparse().indices()
        sparse_values = list()
        for index in sparse_indices.t(): sparse_values += [t[tuple(index)]]
        sparse_values = torch.tensor(sparse_values)
        
        mean = sparse_values.mean()
        std = sparse_values.std()
        
        if (verbose): print(f"Mean: {mean}; Std: {std}")
        del sparse_indices, sparse_values
        
        for index in indices: 
            val = max(mean + ((random.random()*2)-1)*std, 1e-2)
            t[index] = val

        
        del mean, std

        
    # convert values to sparse tensor format
    sparse_indices = (t != -1).to(int).to_sparse().indices()
    sparse_values = list()
    for index in sparse_indices.t(): sparse_values += [t[tuple(index)]]
    sparse_values = torch.tensor(sparse_values)

    sparse_tensor = torch.sparse_coo_tensor(indices = sparse_indices, values= sparse_values, size = t.size()).coalesce()
    
    if (verbose): print(f"\n{len(sparse_tensor.indices().t())}/{int(total_cells)} = {len(sparse_tensor.indices().t())/total_cells:.4f} non-zero values.")


    return sparse_tensor.to(device)



def get_unique(all_indices, train_indices):

    unique_dict = dict()

    for i in [tuple([int(y) for y in x]) for x in all_indices]:
        unique_dict[i] = True

    for j in [tuple([int(y) for y in x]) for x in train_indices]:
        # if (j in unique_dict): del unique_dict[j]
        del unique_dict[j]

    unique_indices = torch.tensor([list(x) for x in list(unique_dict)])

    del unique_dict
    
    return unique_indices

def get_unique_MAE(model, full_t, sparse_t, return_errors = False, return_indices = False, device = "cuda" if torch.cuda.is_available() else "cpu"):

    unique_indices = get_unique(all_indices = get_sparse_tensor(t = full_t.clone(), portion = 1.0).cpu().indices().t(),
                                train_indices = sparse_t.cpu().indices().t()).to(device)
    
    unique_recon = model(unique_indices).cpu()

    if (return_errors): error_list = list()
    
    unique_recon_MAE = 0
    for i in range(len(unique_indices)):
        unique_index = tuple(unique_indices[i])
        unique_value = unique_recon[i]

        current_error = abs(unique_value - full_t[unique_index])
        if (return_errors): 
            if (return_indices): error_list += [(tuple([int(x) for x in unique_index]), float(current_error))]
            else: error_list += [float(current_error)]
        
        unique_recon_MAE += float(abs(current_error))

    unique_recon_MAE /= len(unique_indices)
    
    if (return_errors): return error_list
    return unique_recon_MAE

# get unique MAE for using a random naive method, such as using the mean + random(-1, 1) * std to fill in values
# or using a random sampling of the existing values in the sparse tensor
def get_naive_unique_MAE(sparse_tensor, original_tensor, random_method = 'mean'):

    original_tensor_sparse = get_sparse_tensor(t = original_tensor.clone(), portion = 1.0)
    naive_t = sparse_tensor.to_dense().clone()
    mask = torch.zeros(sparse_tensor.size())

    unique_indices = get_unique(all_indices = original_tensor_sparse.indices().t(),
                            train_indices = sparse_tensor.indices().t())


    if (random_method == 'mean'):
        center = sparse_tensor.values().mean()
        std = sparse_tensor.values().std()
    
    if (random_method == 'median'):
        center = sparse_tensor.values().median()
        std = sparse_tensor.values().std()

    for torch_index in unique_indices:

            index = tuple([int(x) for x in torch_index])
            
            # random values near the center
            if (random_method == 'mean' or random_method == 'median'):
                inferred_value = center + ((random.random()-0.5) * std)
                
            elif (random_method == 'sampling'):
                inferred_value = list(sparse_tensor.values())[int(random.random()*len(sparse_tensor.values()))]
            
            else:
                inferred_value = list(sparse_tensor.values())[int(random.random()*len(sparse_tensor.values()))]
            naive_t[index] = inferred_value
            mask[index] = 1
            
            
    unique_MAE = abs((naive_t - original_tensor) * mask).mean()
    
    return unique_MAE


def get_slice_MAE(model, full_t, sparse_t, slice):
    
    return  torch.tensor([x[1] for x in 
                            get_unique_MAE(model,
                                    full_t = full_t,
                                    sparse_t = sparse_t,
                                    return_errors = True,
                                    return_indices = True)
                            if x[0][0] == slice]).mean()


def rmse_(val, rec):
    '''
    Calculate Root Mean Squared Error (RMSE)
    '''
    return torch.sqrt(torch.mean((val-rec) ** 2))

def nre_(val, rec):
    '''
    Calculate Normalized Reconstruction Error (NRE)
    '''
    a = torch.sqrt(((val-rec) ** 2).sum())
    b = torch.sqrt((val ** 2).sum())
    return a / b