import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.validation import check_random_state


class DotMap:
    pass


class COODataset(Dataset):
    def __init__(self, idxs, vals):
        self.idxs = idxs
        self.vals = vals

    def __len__(self):
        return self.vals.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.vals[idx]


_smooth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Kernel(nn.Module):
    ''' Implement a kernel smoothing regularization'''

    def __init__(self, window, density, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(_smooth_device)

    def gaussian(self):
        ''' Make a Gaussian kernel'''

        window = int(self.window - 1) / 2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window + 1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1).to(torch.double)

    def forward(self, factor):
        ''' Perform a Gaussian kernel smoothing on a temporal factor'''

        row, col = factor.shape
        conv = F.conv2d(factor.view(1, 1, row, col), self.weight,
                        padding=(int((self.window - 1) / 2), 0))
        return conv.view(row, col)


class Inverse_Kernel(nn.Module):
    ''' Implement a kernel smoothing regularization'''

    def __init__(self, window, density, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(_smooth_device)

    def gaussian(self):
        ''' Make a Gaussian kernel'''

        window = int(self.window - 1) / 2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window + 1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1).to(torch.double)

    def forward(self, factor):

        row, col = factor.shape

        factor = factor.T
        conv = F.conv2d(factor.view(1, 1, col, row), self.weight,
                        padding=(int((self.window - 1) / 2), 0))

        return conv.view(row, col)


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
        concat = torch.concat(facs, dim=-1)  # NNZ x rank x nmode
        rec = torch.prod(concat, dim=-1)    # NNZ x ranak
        return rec.sum(-1)

    def forward(self, idxs):
        return self.recon(idxs)


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

        self.smooth = Kernel(self.window, density=None).to(_smooth_device)
        self.inverse_smooth = Inverse_Kernel(self.inverse_window, density=None).to(_smooth_device)

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
        concat = torch.concat(facs, dim=-1)  # NNZ x rank x nmode
        rec = torch.prod(concat, dim=-1)    # NNZ x ranak
        return rec.sum(-1)

    def forward(self, idxs):
        return self.recon(idxs)

    def inverse_std_error(self, mode):
        return self.embeds[mode].weight.std(axis=1).sum()

    def smooth_reg(self, mode):
        ''' Perform a smoothing regularization on the time factor '''

        smoothed = self.smooth(self.embeds[mode].weight)

        sloss = (smoothed - self.embeds[mode].weight).pow(2)

        return sloss.sum()

    def inverse_smooth_reg(self, mode):
        ''' Perform a smoothing regularization on the time factor '''

        smoothed = self.inverse_smooth(self.embeds[mode].weight)

        sloss = (smoothed - self.embeds[mode].weight).pow(2)

        return sloss.sum()


def train_tensor_completion(model_type,
                            sparse_tensor,
                            rank=5,
                            num_epochs=15_000,
                            batch_size=256,
                            lr=5e-3,
                            wd=5e-4,
                            loss_p=2,
                            zero_lambda=1,
                            cpd_smooth_lambda=2,
                            cpd_smooth_window=3,
                            cpd_inverse_smooth_lambda=0,
                            cpd_inverse_smooth_window=3,
                            cpd_inverse_std_lambda=0,
                            non_smooth_modes=list(),
                            non_inverse_smooth_modes=list(),
                            tucker_in_drop=0.1,
                            tucker_hidden_drop=0.1,
                            train_norm=None,
                            early_stopping=True,
                            flags=15,
                            verbose=False,
                            epoch_display_rate=1,
                            val_size=0.2,
                            return_errors=False,
                            reinitialize_count=0,
                            convert_to_cpd=False,
                            for_queries=False,
                            device="cuda" if torch.cuda.is_available() else "cpu"):

    if (verbose): print(f"Rank = {rank}; lr = {lr}; wd = {wd}\n")

    train_indices = sparse_tensor.indices().t()
    train_values = sparse_tensor.values()
    tensor_size = sparse_tensor.size()

    training_indices = train_indices.to(device)  # NNZ x mode
    training_values = train_values.to(device)    # NNZ
    training_values = training_values.to(torch.double)

    indices, val_indices, values, val_values = train_test_split(training_indices, training_values, test_size=val_size, random_state=18)

    cfg = DotMap()

    cfg.norm = lambda x: x
    cfg.unnorm = lambda x: x

    if train_norm is not None:

        if train_norm == 'minmax':

            max_ = training_values.max()
            min_ = training_values.min()

            cfg.norm = lambda x: (x - min_) / (max_ - min_)
            cfg.unnorm = lambda x: (x * (max_ - min_)) + min_

        elif train_norm == 'standard':

            mean_ = training_values.mean()
            std_ = training_values.std()

            cfg.norm = lambda x: (x - mean_) / (std_)
            cfg.unnorm = lambda x: (x * std_) + mean_

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

    else:
        print("No Model Selected!")
        model = CPD(cfg).to(device)

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

            loss.backward()
            optimizer.step()

        model.eval()
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                train_rec = model(indices)
                train_MAE = abs(train_rec - values).mean()

                val_rec = model(val_indices)
                val_MAE = abs(val_rec - val_values).mean()

                # for early stopping
                if (early_stopping):

                    if (old_MAE < val_MAE):
                        flag += 1

                    if flag == flags:
                        break

                    if (old_MAE == val_MAE):
                        flag_2 += 1

                if flag_2 == 25:
                    break

                old_MAE = val_MAE

                err_list += [old_MAE]

                if (verbose and ((epoch + 1) % epoch_display_rate == 0)):
                    print(f"Epoch {epoch+1} Train_MAE: {train_MAE:.4f} Val_MAE: {val_MAE:.4f}\t")

    if (verbose): print()

    # reinitialize model if it didn't converge!
    if (torch.tensor(err_list[10:]).std() < 1e-6):

        if (reinitialize_count >= 5) and convert_to_cpd:

            if (verbose): print(f"\nConverting {model_type} to cpd!\n")

            return train_tensor_completion(model_type='cpd',
                                            sparse_tensor=sparse_tensor,
                                            rank=rank,
                                            num_epochs=num_epochs,
                                            batch_size=batch_size,
                                            lr=lr,
                                            wd=wd,
                                            tucker_in_drop=tucker_in_drop,
                                            tucker_hidden_drop=tucker_hidden_drop,
                                            early_stopping=early_stopping,
                                            flags=flags,
                                            verbose=verbose,
                                            epoch_display_rate=epoch_display_rate,
                                            val_size=val_size,
                                            return_errors=return_errors,
                                            reinitialize_count=reinitialize_count + 1)

        if (verbose): print(f"\nReinitializing {model_type}! Reinitialize Count: {reinitialize_count}.\n")

        return train_tensor_completion(model_type=model_type,
                                       sparse_tensor=sparse_tensor,
                                       rank=rank,
                                       num_epochs=num_epochs,
                                       batch_size=batch_size,
                                       lr=lr,
                                       wd=wd,
                                       tucker_in_drop=tucker_in_drop,
                                       tucker_hidden_drop=tucker_hidden_drop,
                                       early_stopping=early_stopping,
                                       flags=flags,
                                       verbose=verbose,
                                       epoch_display_rate=epoch_display_rate,
                                       val_size=val_size,
                                       return_errors=return_errors,
                                       reinitialize_count=reinitialize_count + 1)

    if (return_errors): return model, torch.tensor(err_list)
    return model


def return_eval(model, x, y, metric='f1', n_splits=5, change_input=None, smote_train=False, random_state=18):

    if change_input is not None:
        mapping = change_input
        mapping_tensor = torch.tensor([mapping[i] for i in range(1, max(mapping.keys()) + 1)])
        x = mapping_tensor[x - 1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    overall_metric = 0
    for i, (train_index, test_index) in enumerate(kf.split(x)):

        X_train = x[train_index]
        X_test = x[test_index]

        Y_train = y[train_index]
        Y_test = y[test_index]

        if (smote_train):
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        if metric == 'f1':

            if (len(set(y)) == 2): average = 'binary'
            else: average = 'weighted'

            metric_value = f1_score(Y_test, preds, average=average)

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
            metric_value = ((Y_test - preds) ** 2).mean()

        elif metric == 'r2':
            metric_value = r2_score(Y_test, preds)

        else:

            print("No metric value for return_eval() !")
            return 0

        overall_metric += metric_value

    return (overall_metric / n_splits)


rand_index = lambda x: tuple([int(random.random() * i) for i in x])


def get_rand_indices(shape, num_indices):

    index_bank = dict()

    while (len(index_bank) < num_indices):
        index_bank[rand_index(shape)] = True

    return list(index_bank)


def return_best_k_params(model,
                         param_dict,
                         X, Y,
                         num_top_combinations=5,
                         cv_splits=5,
                         tensor_portion=0.1,
                         tensor_completion_model='costco',
                         rank=25,
                         metric='f1',
                         device='cpu',
                         verbose=False):

    '''
    model : sklearn model to optimize
    param_dict : dictionary for parameter space to explore, single value for fixed parameter, list for a range to explore
    num_top_combinations : number of predicted optimal combinations to return
    cv_splits : number of splits for k-fold cross validation evaluation
    tensor_portion : fraction/number of hyperparameter combinations for the tensor completion to train on (out of the entire hyperparameter space)
    tensor_completion_model : tensor completion model
    rank : tensor completion rank decomposition
    metric : evaluation metric to optimize
    device : 'cuda' or 'cpu'
    verbose : True or False
    '''

    param_list = list(param_dict)

    tensor_size = [len(param_dict[x]) for x in param_dict]

    total_cells = 1
    for s in tensor_size: total_cells *= s

    if tensor_portion < 1:
        portion_of_combinations = tensor_portion
    else:
        portion_of_combinations = tensor_portion / total_cells

    num_indices = int(total_cells * portion_of_combinations)

    if (verbose): print(f"{num_indices}/{total_cells} total combinations in sparse tensor.")

    tensor_indices = get_rand_indices(shape=tensor_size, num_indices=num_indices)

    param_combinations = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in tensor_indices]

    values = list()

    it = 0
    for param_combination in param_combinations:

        model.set_params(**param_combination)

        value = return_eval(model=model, x=X, y=Y, metric=metric, n_splits=cv_splits, smote_train=False, random_state=18)

        values += [value]

        it += 1
        if (verbose): print(f"{it}/{len(param_combinations)} param_combinations done.")

    values = torch.tensor(values)

    sparse_tensor = torch.sparse_coo_tensor(indices=torch.tensor(tensor_indices).t(), values=values, size=tensor_size).coalesce().to(device)

    if (verbose): print("\nRunning sparse tensor completion...")

    STC_model = train_tensor_completion(model_type=tensor_completion_model,
                                        sparse_tensor=sparse_tensor,
                                        rank=rank,
                                        num_epochs=15000,
                                        batch_size=256,
                                        lr=5e-3,
                                        wd=1e-4,
                                        tucker_in_drop=0.1,
                                        tucker_hidden_drop=0.1,
                                        early_stopping=True,
                                        flags=15,
                                        verbose=False,
                                        epoch_display_rate=1,
                                        val_size=0.2,
                                        convert_to_cpd=True,
                                        device=device)

    grid = torch.meshgrid(*[torch.arange(s) for s in tensor_size], indexing='ij')
    all_indices = torch.stack(grid, dim=-1).reshape(-1, len(tensor_size))

    # List of indices to exclude
    exclude_tensor = sparse_tensor.indices().t().clone().to('cpu')

    # Create a boolean mask
    mask = ~(all_indices.unsqueeze(1) == exclude_tensor.unsqueeze(0)).all(dim=2).any(dim=1)

    # Filter the indices
    unique_indices = all_indices[mask]

    del all_indices, mask, grid, exclude_tensor

    inferred_values = STC_model(unique_indices.to(device))

    dense_tensor_values = torch.concat((values, inferred_values.to('cpu')))
    dense_tensor_indices = torch.concat((sparse_tensor.indices().t().to('cpu'), unique_indices))

    dense_tensor = torch.sparse_coo_tensor(indices=dense_tensor_indices.t(), values=dense_tensor_values, size=tensor_size).coalesce()
    tensor = dense_tensor.to_dense()

    del dense_tensor_values, dense_tensor_indices, inferred_values, sparse_tensor, unique_indices, values, dense_tensor

    if (verbose): print("Done with sparse tensor completion!")

    largest_value = metric in ['f1', 'f1-score', 'f1_score', 'precision', 'recall', 'accuracy', 'auroc', 'r2']

    values, indices = torch.topk(tensor.flatten(), num_top_combinations, largest=largest_value)

    top_k_indices = np.array(np.unravel_index(indices.numpy(), tensor.shape)).T

    best_params = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in top_k_indices]

    if (verbose): print("\nEvaluating predicted best parameters.")

    best_estimated_params = list()

    for i in range(len(best_params)):

        parameters = best_params[i]

        model.set_params(**parameters)

        actual_eval = return_eval(model=model, x=X, y=Y, metric=metric, n_splits=cv_splits, smote_train=False, random_state=18)
        predicted_eval = values[i]

        best_estimated_params += [(parameters, float(actual_eval))]

    best_estimated_params.sort(key=lambda x: x[-1], reverse=True)

    print("Done!")

    return best_estimated_params
