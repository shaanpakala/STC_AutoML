
from notebooks.utilities.helper_functions import *

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import chain as chain_generators

from torch.nn import MSELoss, L1Loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class CNN(nn.Module):
    def __init__(self, 
                 input_dim = None, 
                 hidden_channels = [100], 
                 kernel_size = 3,
                 output_dim = 2,
                 activation = 'relu',
                 last_act = 'softmax'
                 ):
        
        super(CNN, self).__init__()
        
        if task_type is None or input_dim is None:
            print('Need task_type & input_dim!')
            return None        
        
        self.conv_dims = [input_dim] + hidden_channels + [1 if task_type == 'regression' else output_dim]
                
        self.fcs = list()
        for i in range(len(self.conv_dims) - 1):
            self.conv += [nn.Conv2d(self.conv_dims[i], self.conv_dims[i+1], 
                                    kernel_size = (kernel_size, kernel_size), padding = int(kernel_size/2))]
                
                
            
        if activation is None: self.act = lambda x: x       
        elif activation == 'relu': self.act = nn.ReLU()
        elif activation == 'tanh': self.act = nn.Tanh()
        elif activation == 'sigmoid': self.act = nn.Sigmoid()
        elif activation == 'softmax': self.act = nn.Softmax(dim = 1)
        else: self.act = lambda x: x

        
        if last_act is None: self.act_out = lambda x: x
        elif last_act == 'softmax': self.act_out = nn.Softmax(dim = 1)
        elif last_act == 'sigmoid': self.act_out = nn.Sigmoid()
        elif last_act == 'tanh': self.act_out = nn.Tanh()
        else: self.act_out = lambda x: x
        
        self.acts = ([self.act] * (len(self.conv)-1)) + [self.act_out]
        
    def forward(self, x):        

        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.acts[i](x)
        
        return x
    
    def predict(self, predict_x):
        return self(predict_x).argmax(dim = 1)
    
    def predict_proba(self, predict_x):
        return self(predict_x).detach().numpy()
    
    
def NNfit(model,
          fit_x, 
          fit_y,
          batch_size = 128,
          num_epochs = 10,
          stop_early = True,
          lr = 1e-3, 
          wd = 1e-4,
          loss_type = 'cross_entropy',
          verbose = False,
          device = 'cpu'):
    
    if loss_type == 'cross_entropy':
        fit_y = fit_y.long()
            
    train_x, x_val, train_y, y_val = train_test_split(fit_x.to(device), fit_y.to(device), test_size = 0.1, shuffle = True, random_state = 18)
    
    train = list()
    for i in range(len(train_x)):
        train.append((train_x[i], train_y[i]))

    trainloader = DataLoader(train, batch_size)

    if loss_type == 'cross_entropy': loss_fn = nn.CrossEntropyLoss() 
    elif loss_type == 'mae': loss_fn = L1Loss() 
    elif loss_type == 'mse': loss_fn = MSELoss() 
    else: loss_fn = nn.CrossEntropyLoss() 
    
    opt = Adam(params = chain_generators(*[fc.parameters() for fc in model.fcs]), 
                lr = lr, 
                weight_decay = wd)  
      
    # opt = Adam(params = model.parameters(), lr = lr, weight_decay = wd)    
        
    flag = 0
    prev_loss = 5e6
    for epoch in range(1, num_epochs+1):

        model.train()
        for batch in trainloader:

            opt.zero_grad()

            x, y = batch 
            Yhat = model(x.float())
                
            loss = loss_fn(Yhat.to('cpu'), y.to('cpu'))

            #backpropogation

            loss.backward()
            opt.step()    
            
        model.eval()
        with torch.no_grad():
            val_rec = model(x_val.float())
            val_loss = loss_fn(val_rec.detach().to('cpu'), y_val.to('cpu'))
            
            if verbose: print(f'Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}')
            
            if prev_loss < val_loss: flag +=1
            if flag >= 5 and stop_early: 
                if verbose: print("Ending training early!")
                break
                
            prev_loss = val_loss + 0
            
        
    return model

def return_nn_eval(x = None, y = None,
                   metric = 'mae',
                   n_splits = None,
                   training_values = 100,
                   num_tests = 3,
                   testing_values = None,
                   hidden_dims = (128),
                   activation = 'relu',
                   last_act = 'softmax',
                   task_type = 'regression',
                   num_classes = None,
                   batch_size = 128,
                   num_epochs = 10,
                   stop_early = True,
                   lr = 1e-2,
                   wd = 1e-3,
                   loss_type = 'mae',
                   verbose = False,
                   smote_train = False,
                   change_input = None,
                   device = 'cpu'):
    
    if x is None or y is None or metric is None or loss_type is None:
        print('Need X, Y, loss_type, and metric to evaluate!')
        return None
    
    
    def post_train_split(model, X_train, Y_train, metric = 'f1'):
        
        if (smote_train):
            smote = SMOTE(random_state=random_state)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
        
        model = NNfit(model = model,
                      fit_x = X_train, 
                      fit_y = Y_train,
                      batch_size = batch_size,
                      num_epochs = num_epochs,
                      stop_early = stop_early,
                      lr = lr,
                      wd = wd,
                      loss_type = loss_type,
                      verbose = verbose,
                      device = device)
        
        preds = model(X_test.float().to(device)).detach().to('cpu')
        
        # _______ insert evaluation metric ___________________________________
        
        if metric == 'f1':
            
            if (num_classes == 2): average = 'binary'
            else: average = 'weighted'
                            
            metric_value = f1_score(Y_test, preds.argmax(dim = 1), 
                                    average = average)
            
        elif metric == 'accuracy':
            metric_value = accuracy_score(Y_test, preds.argmax(dim = 1))
            
        elif metric == 'precision':
            metric_value = precision_score(Y_test, preds.argmax(dim = 1))
            
        elif metric == 'recall':
            metric_value = recall_score(Y_test, preds.argmax(dim = 1))
            
        elif metric == 'auroc':
            metric_value = roc_auc_score(Y_test, preds[:, 1])
        
            
        elif metric == 'mae':
            metric_value = abs(Y_test - preds).mean()
            
        elif metric == 'mse':
            metric_value = ((Y_test - preds)**2).mean()
            
        else:
            
            print("No metric value for return_eval() !")
            return 0
        
        # ____________________________________________________________________
        
        return metric_value
    
    if change_input is not None:
        mapping = change_input
        mapping_tensor = torch.tensor([mapping[i] for i in range(1, max(mapping.keys()) + 1)])
        x = mapping_tensor[x - 1]
        
    if len(x.shape) > 2:
        x = x.flatten(start_dim = 1)
    
    overall_metric = 0
    if n_splits is None:
        
        indices = [i for i in range(len(y))]
        random.shuffle(indices)
        
        start_i = 0
        stop_i = training_values
        
        for test in range(num_tests):
            
            curr_model = CNN(task_type = task_type,
                             input_dim = x.shape[-1],
                             hidden_dims = list(hidden_dims),
                             activation = activation,
                             last_act = last_act,
                             output_dim = num_classes).to(device)
            
            curr_indices = {i:True for i in indices[start_i:stop_i]}

            X_train = torch.tensor(np.array([x[z] for z in range(len(x)) if z in curr_indices]))
            Y_train = torch.tensor(np.array([y[z] for z in range(len(x)) if z in curr_indices]))
            
            X_test = torch.tensor(np.array([x[z] for z in range(len(x)) if z not in curr_indices]))
            Y_test = torch.tensor(np.array([y[z] for z in range(len(x)) if z not in curr_indices]))
            
            if testing_values is not None:
                X_test, Y_test = X_test[:testing_values], Y_test[:testing_values]
            
            start_i, stop_i = start_i + training_values, stop_i + training_values
        
            overall_metric += post_train_split(model = curr_model,
                                               X_train = X_train,
                                               Y_train = Y_train,
                                               metric = metric)
            
            del curr_model, curr_indices, X_train, Y_train, X_test, Y_test
            

        return (overall_metric/num_tests)
        
    
    else:
        
        
        kf = KFold(n_splits=n_splits, shuffle = True, random_state = random_state)
            
        for i, (train_index, test_index) in enumerate(kf.split(x)):

            curr_model = CNN(task_type = task_type,
                             input_dim = x.shape[-1],
                             hidden_dims = list(hidden_dims),
                             activation = activation,
                             last_act = last_act,
                             output_dim = num_classes).to(device)
            
            X_train = x[train_index]
            X_test = x[test_index]
            
            Y_train = y[train_index]
            Y_test = y[test_index]
            
            overall_metric += post_train_split(model = curr_model,
                                               X_train = X_train,
                                               Y_train = Y_train,
                                               metric = metric)
            
            del curr_model, X_train, Y_train, X_test, Y_test
            
        return (overall_metric/n_splits)



def return_best_k_params(X = None, Y = None,
                         num_top_combinations = 5,
                         tensor_completion_model = 'cpd.smooth',
                         rank = 10,
                         task_type = 'regression',
                         metric = 'mae',
                         tensor_training_portion = None,
                         tensor_training_values = 100,
                         hyperameter_dict = None,
                         cv_splits = None,
                         training_values = 100,
                         num_tests = 3,
                         testing_values = None,
                         verbose = False,
                         device = 'cpu'):
    
    if X is None or Y is None or hyperameter_dict is None:
        print('Need X, Y, and param_dict to evaluate!')
        return None
    
    param_dict = {param:hyperameter_dict[param] for param in list(hyperameter_dict) if type(hyperameter_dict[param]) == type([0])}
    constant_param_dict = {param:hyperameter_dict[param] for param in list(hyperameter_dict) if type(hyperameter_dict[param]) != type([0])}

    param_list = list(param_dict)

    tensor_size = [len(param_dict[x]) for x in param_dict]

    total_cells = 1
    for s in tensor_size: total_cells*=s
    
    if tensor_training_portion is None and tensor_training_values is None:
        return None
    elif tensor_training_values is None:
        portion_of_combinations = tensor_training_portion
    else:
        portion_of_combinations = tensor_training_values/total_cells

    num_indices = int(total_cells*portion_of_combinations)

    if (verbose): print(f"{num_indices}/{total_cells} total combinations in sparse tensor.")

    tensor_indices = get_rand_indices(shape = tensor_size, num_indices = num_indices)

    param_combinations = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in tensor_indices]

    values = list()
    
    if task_type == 'classification': num_classes = len(set(list(Y.squeeze())))
    else: num_classes = None
    
    default_params = {'hidden_dims':(128),
                      'activation':'relu',
                      'last_act':None,
                      'batch_size':128,
                      'num_epochs':10,
                      'stop_early':True,
                      'lr':1e-2,
                      'wd':1e-3,
                      'loss_fn':'mae' if task_type == 'regression' else 'cross_entropy',
                      'smote_train':False,
                      'change_input':None}

    it = 0
    for param_combination in param_combinations:

        curr_params = default_params.copy()
        
        for p in list(default_params):
            
            if p in constant_param_dict: curr_params[p] = constant_param_dict[p]
            if p in param_combination: curr_params[p] = param_combination[p]
                
        
        value = return_nn_eval(x = X, y = Y,
                               metric = metric,
                               n_splits = None,
                               training_values = training_values,
                               num_tests = num_tests,
                               testing_values = testing_values,
                               hidden_dims = curr_params['hidden_dims'],
                               activation = curr_params['activation'],
                               last_act = curr_params['last_act'],
                               task_type = task_type,
                               num_classes = num_classes,
                               batch_size = curr_params['batch_size'],
                               num_epochs = curr_params['num_epochs'],
                               stop_early = curr_params['stop_early'],
                               lr = curr_params['lr'],
                               wd = curr_params['wd'],
                               loss_type = curr_params['loss_fn'],
                               verbose = False,
                               smote_train = curr_params['smote_train'],
                               change_input = curr_params['change_input'],
                               device = device)

        values += [value]
        
        it+=1
        if (verbose): print(f"{it}/{len(param_combinations)} param_combinations done.")

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
                
        
        actual_eval = return_nn_eval(x = X, y = Y,
                               metric = metric,
                               n_splits = None,
                               training_values = training_values,
                               num_tests = num_tests,
                               testing_values = testing_values,
                               hidden_dims = curr_params['hidden_dims'],
                               activation = curr_params['activation'],
                               last_act = curr_params['last_act'],
                               task_type = task_type,
                               num_classes = num_classes,
                               batch_size = curr_params['batch_size'],
                               num_epochs = curr_params['num_epochs'],
                               stop_early = curr_params['stop_early'],
                               lr = curr_params['lr'],
                               wd = curr_params['wd'],
                               loss_type = curr_params['loss_fn'],
                               verbose = False,
                               smote_train = curr_params['smote_train'],
                               change_input = curr_params['change_input'],
                               device = device)
        
        predicted_eval = values[i]
                
        # best_estimated_params += [(parameters, float(predicted_eval), float(actual_eval))]
        best_estimated_params += [(parameters, float(actual_eval))]
        
        
    best_estimated_params.sort(key = lambda x: x[-1], reverse = True)
    
    print("Done!")
    
    return best_estimated_params

    
    
    
    