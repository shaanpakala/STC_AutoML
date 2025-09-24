
from notebooks.utilities.helper_functions import *

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import chain as chain_generators

from torch.nn import MSELoss, L1Loss


def return_tc_eval(x = None, y = None,
                   task_tensor_size = None,
                   metric = 'mae',
                   n_splits = None,
                   training_values = 100,
                   num_tests = 3,
                   testing_values = None,
                   tensor_completion_model = 'cpd',
                   rank = 10,
                   batch_size = 128,
                   num_epochs = 10,
                   stop_early = True,
                   lr = 1e-2,
                   wd = 1e-3,
                   loss_p = 2,
                   cpd_smooth_lambda = 2,
                   cpd_smooth_window = 3,
                   NeAT_drop = 0.1,
                   NeAT_drop2 = 0.5,
                   NeAT_hidden_dim = 32,
                   cpd_inverse_smooth_lambda = 0,
                   cpd_inverse_smooth_window = 3,
                   non_smooth_modes = list(),
                   non_inverse_smooth_modes = list(),
                   cpd_inverse_std_lambda = 0,
                   zero_lambda = 1,
                   verbose = False,
                   device = 'cpu'
                   ):
    
    if x is None or y is None or task_tensor_size is None or metric is None:
        print('Need X, Y, task_tensor_size, and metric to evaluate!')
        return None
    
    
    def post_train_split(sparse_tensor, metric = 'mae'):

        model = train_tensor_completion(model_type = tensor_completion_model,
                                        sparse_tensor = sparse_tensor,
                                        rank = rank,
                                        num_epochs = num_epochs,
                                        batch_size = batch_size,
                                        lr = lr,
                                        wd = wd,
                                        loss_p = loss_p,
                                        cpd_smooth_lambda = cpd_smooth_lambda,
                                        cpd_smooth_window = cpd_smooth_window,
                                        non_smooth_modes = non_smooth_modes,
                                        non_inverse_smooth_modes = non_inverse_smooth_modes,
                                        NeAT_drop = NeAT_drop,
                                        NeAT_drop2 = NeAT_drop2,
                                        NeAT_hidden_dim = NeAT_hidden_dim,
                                        cpd_inverse_smooth_lambda = cpd_inverse_smooth_lambda,
                                        cpd_inverse_smooth_window = cpd_inverse_smooth_window,
                                        cpd_inverse_std_lambda = cpd_inverse_std_lambda,
                                        zero_lambda = zero_lambda,
                                        early_stopping = stop_early,
                                        device = device,
                                        verbose = verbose)
        
        preds = model(X_test.to(device)).detach().to('cpu')
        
        # _______ insert evaluation metric ___________________________________
        
            
        if metric == 'mae':
            
            metric_value = abs(Y_test - preds).mean()
            
        elif metric == 'mse':
            
            metric_value = ((Y_test - preds)**2).mean()
            
        else:
            
            print("No metric value for return_eval() !")
            return 0
        
        # ____________________________________________________________________
        
        return metric_value
    
    
    overall_metric = 0
    if n_splits is None:
        
        indices = [i for i in range(len(y))]
        random.shuffle(indices)
        
        start_i = 0
        stop_i = training_values
        
        for test in range(num_tests):
            
            curr_indices = {i:True for i in indices[start_i:stop_i]}

            X_train = torch.tensor(np.array([x[z] for z in range(len(x)) if z in curr_indices]))
            Y_train = torch.tensor(np.array([y[z] for z in range(len(x)) if z in curr_indices]))
            
            X_test = torch.tensor(np.array([x[z] for z in range(len(x)) if z not in curr_indices]))
            Y_test = torch.tensor(np.array([y[z] for z in range(len(x)) if z not in curr_indices]))
            
            sparse_tensor = torch.sparse_coo_tensor(indices = X_train.t(), 
                                                    values = Y_train, 
                                                    size = task_tensor_size).coalesce()
            
            if testing_values is not None:
                X_test, Y_test = X_test[:testing_values], Y_test[:testing_values]
            
            start_i, stop_i = start_i + training_values, stop_i + training_values
        
            overall_metric += post_train_split(sparse_tensor = sparse_tensor,
                                               metric = metric)
            
            del curr_indices, X_train, Y_train, X_test, Y_test
            

        return (overall_metric/num_tests)
        
    
    else:
        
        
        kf = KFold(n_splits=n_splits, shuffle = True, random_state = random_state)
            
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            
            X_train = torch.tensor(x[train_index])
            X_test = torch.tensor(x[test_index])
            
            Y_train = torch.tensor(y[train_index])
            Y_test = torch.tensor(y[test_index])
            
            
            sparse_tensor = torch.sparse_coo_tensor(indices = X_train.t(), 
                                                    values = Y_train, 
                                                    size = task_tensor_size).coalesce()
            
            overall_metric += post_train_split(sparse_tensor = sparse_tensor,
                                               metric = metric)
            
            del X_train, Y_train, X_test, Y_test
            
        return (overall_metric/n_splits)



def return_best_k_params(X = None, Y = None, task_tensor_size = None,
                         num_top_combinations = 5,
                         tensor_completion_model = 'cpd.smooth',
                         rank = 6,
                         metric = 'mae',
                         tensor_training_portion = None,
                         tensor_training_values = None,
                         hyperameter_dict = None,
                         cv_splits = None,
                         training_values = 100,
                         num_tests = 3,
                         testing_values = None,
                         verbose = False,
                         device = 'cpu'):
        
    
    if X is None or Y is None or task_tensor_size is None or hyperameter_dict is None:
        print('Need X, Y, tensor_size, and param_dict to evaluate!')
        return None
    
    param_dict = {param:hyperameter_dict[param] for param in list(hyperameter_dict) if type(hyperameter_dict[param]) == type([0])}
    constant_param_dict = {param:hyperameter_dict[param] for param in list(hyperameter_dict) if type(hyperameter_dict[param]) != type([0])}

    param_list = list(param_dict)

    tensor_size = [len(param_dict[x]) for x in param_dict]

    total_cells = 1
    for s in tensor_size: total_cells*=s

        
    if tensor_training_portion is None and tensor_training_values is None:
        portion_of_combinations = 100
    elif tensor_training_values is None:
        portion_of_combinations = tensor_training_portion
    else:
        portion_of_combinations = tensor_training_values/total_cells

    num_indices = int(total_cells*portion_of_combinations)

    if (verbose): print(f"{num_indices}/{total_cells} total combinations in sparse tensor.\n")

    tensor_indices = get_rand_indices(shape = tensor_size, num_indices = num_indices)

    param_combinations = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in tensor_indices]

    values = list()
    
    default_params = {
                      'tensor_model':'cpd',
                      'rank':6,
                      'batch_size':128,
                      'num_epochs':10,
                      'stop_early':True,
                      'lr':1e-2,
                      'wd':1e-3,
                      'loss_p':2,
                      'cpd_smooth_lambda':2,
                      'cpd_smooth_window':3,
                      'non_smooth_modes':list(),
                      'NeAT_drop1':0.1,
                      'NeAT_drop2':0.5,
                      'NeAT_hidden_dim':32,
                      'cpd_inverse_smooth_lambda':0,
                      'cpd_inverse_smooth_window':3,
                      'cpd_inverse_std_lambda':0,
                      'non_inverse_smooth_modes':list(),
                      'zero_lambda':1
                      }

    it = 0
    for param_combination in param_combinations:

        curr_params = default_params.copy()
        
        for p in list(default_params):
            
            if p in constant_param_dict: curr_params[p] = constant_param_dict[p]
            if p in param_combination: curr_params[p] = param_combination[p]
                        
        value = return_tc_eval(x = X, y = Y,
                               task_tensor_size = task_tensor_size,
                               metric = metric,
                               n_splits = cv_splits,
                               training_values = training_values,
                               num_tests = num_tests,
                               testing_values = testing_values,
                               tensor_completion_model = curr_params['tensor_model'],
                               rank = curr_params['rank'],
                               batch_size = curr_params['batch_size'],
                               num_epochs = curr_params['num_epochs'],
                               stop_early = curr_params['stop_early'],
                               lr = curr_params['lr'],
                               wd = curr_params['wd'],
                               loss_p = curr_params['loss_p'],
                               cpd_smooth_lambda = curr_params['cpd_smooth_lambda'],
                               cpd_smooth_window = curr_params['cpd_smooth_window'],
                               non_smooth_modes = curr_params['non_smooth_modes'],
                               NeAT_drop = curr_params['NeAT_drop1'],
                               NeAT_drop2 = curr_params['NeAT_drop2'],
                               NeAT_hidden_dim = curr_params['NeAT_hidden_dim'],
                               cpd_inverse_smooth_lambda = curr_params['cpd_inverse_smooth_lambda'],
                               cpd_inverse_smooth_window = curr_params['cpd_inverse_smooth_window'],
                               cpd_inverse_std_lambda = curr_params['cpd_inverse_std_lambda'],
                               non_inverse_smooth_modes = curr_params['non_inverse_smooth_modes'],
                               zero_lambda = curr_params['zero_lambda'],
                               verbose = False,
                               device = device)

        values += [value]
        
        it+=1
        if (verbose): 
            print(f"({it}/{len(param_combinations)}) param_combinations done.", end = " ")
            if verbose == 2: print(f"{curr_params}")
            else: print()

    values = torch.tensor(values)

    sparse_tensor = torch.sparse_coo_tensor(indices = torch.tensor(tensor_indices).t(), values= values, size = tensor_size).coalesce().to(device)
        
    if (verbose): print("\nRunning sparse tensor completion...")
    
    STC_model = train_tensor_completion(model_type = tensor_completion_model, 
                                        sparse_tensor = sparse_tensor,
                                        rank = rank, 
                                        num_epochs = 15_000, 
                                        batch_size = 256, 
                                        lr=5e-3, 
                                        wd=1e-4, 
                                        tucker_in_drop = 0.1,
                                        tucker_hidden_drop = 0.1,
                                        early_stopping = True, 
                                        flags = 15, 
                                        verbose = False, 
                                        epoch_display_rate = 1, 
                                        val_size = 0.2,
                                        convert_to_cpd = True,
                                        for_queries = False,
                                        device = device)


    # asked perplexity.ai for help with this

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

    dense_tensor = torch.sparse_coo_tensor(indices = dense_tensor_indices.t(), values = dense_tensor_values, size = tensor_size).coalesce()
    tensor = dense_tensor.to_dense()

    del dense_tensor_values, dense_tensor_indices, inferred_values, sparse_tensor, unique_indices, values, dense_tensor

    if (verbose): print("Done with sparse tensor completion!")


    largest_value = metric in ['f1']
    # asked perplexity.ai for help with this
    values, indices = torch.topk(tensor.flatten(), num_top_combinations, largest = largest_value)
    
    # Convert flat indices back to 3D indices
    top_k_indices = np.array(np.unravel_index(indices.numpy(), tensor.shape)).T
        
        
    best_params = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in top_k_indices]
    
    
    if (verbose): print("\nEvaluating predicted best parameters.")
    
    best_estimated_params = list()


    for i in range(len(best_params)):
        
        parameters = best_params[i]
        curr_params = default_params.copy()
        
        for p in list(default_params):
            
            if p in constant_param_dict: curr_params[p] = constant_param_dict[p]
            if p in parameters: curr_params[p] = parameters[p]
        
        actual_eval = return_tc_eval(x = X, y = Y,
                                     task_tensor_size = task_tensor_size,
                                     metric = metric,
                                     n_splits = cv_splits,
                                     training_values = training_values,
                                     num_tests = num_tests,
                                     testing_values = testing_values,
                                     tensor_completion_model = curr_params['tensor_model'],
                                     rank = curr_params['rank'],
                                     batch_size = curr_params['batch_size'],
                                     num_epochs = curr_params['num_epochs'],
                                     stop_early = curr_params['stop_early'],
                                     lr = curr_params['lr'],
                                     wd = curr_params['wd'],
                                     loss_p = curr_params['loss_p'],
                                     cpd_smooth_lambda = curr_params['cpd_smooth_lambda'],
                                     cpd_smooth_window = curr_params['cpd_smooth_window'],
                                     non_smooth_modes = curr_params['non_smooth_modes'],
                                     NeAT_drop = curr_params['NeAT_drop1'],
                                     NeAT_drop2 = curr_params['NeAT_drop2'],
                                     NeAT_hidden_dim = curr_params['NeAT_hidden_dim'],
                                     cpd_inverse_smooth_lambda = curr_params['cpd_inverse_smooth_lambda'],
                                     cpd_inverse_smooth_window = curr_params['cpd_inverse_smooth_window'],
                                     cpd_inverse_std_lambda = curr_params['cpd_inverse_std_lambda'],
                                     non_inverse_smooth_modes = curr_params['non_inverse_smooth_modes'],
                                     zero_lambda = curr_params['zero_lambda'],
                                     verbose = False,
                                     device = device)

        if verbose: 
            print(f"({i+1}/{len(best_params)}) Actual Evaluation = {actual_eval:.4f}.", end = " ")
            if verbose > 1: print(f"Params: {curr_params}.")
            else: print()

        predicted_eval = values[i]

        # best_estimated_params += [(parameters, float(predicted_eval), float(actual_eval))]
        best_estimated_params += [(parameters, float(actual_eval))]

    best_estimated_params.sort(key = lambda x: x[-1], reverse = True)

    print("Done!")

    return best_estimated_params