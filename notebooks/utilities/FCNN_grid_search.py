from notebooks.utilities.helper_functions import *

from torch import nn
from torch.optim import Adam


class FCNN_clf(nn.Module):
    def __init__(self, input_dim, param_dict, num_classes = 2):
        super(FCNN_clf, self).__init__()
        
        self.act_layer = None
        self.last_act_layer = None
        self.layer_type = None
        
        
        for param, value in param_dict.items():
            setattr(self, param, value)      
         
        self.in_ = nn.Linear(input_dim, self.hidden_dim)
        
        if (self.layer_type is None):
            self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        elif (self.layer_type == 'linear'):
            self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        # elif (self.layer_type == 'conv1d'):
        #     self.hidden_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
                
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        
        if (self.act_layer is not None):
            
            if (self.act_layer == 'relu'):
                self.act = nn.ReLU()
                
            elif (self.act_layer == 'tanh'):
                self.act = nn.Tanh()
                
            elif (self.act_layer == 'sigmoid'):
                self.act = nn.Sigmoid()
            
            
        if (self.last_act_layer is not None):
            
            if (self.last_act_layer == 'relu'):
                self.last_act = nn.ReLU()
                
            elif (self.last_act_layer == 'tanh'):
                self.last_act = nn.Tanh()
                
            elif (self.last_act_layer == 'sigmoid'):
                self.last_act = nn.Sigmoid()
        
        self.out_ = nn.Linear(self.hidden_dim, num_classes)
        
        
    def forward(self, x):
        
        x = self.in_(x)
        
        for i in range(self.num_layers):
            x = self.hidden_layer(x)
            
            if (self.act_layer is not None):
                x = self.act(x)
                
            x = self.dropout(x)
            
        x = self.out_(x)
                
        if (self.last_act_layer is not None):
            x = self.last_act(x)
        
        return x
    
    
    
    
def train_block(clf, opt, loss_fn, X_train, Y_train, num_epochs = 10, batch_size = 128, verbose = False):

    train = list()
    for i in range(len(X_train)):
        train.append((X_train[i], Y_train[i]))

    trainloader = DataLoader(train, batch_size)

    for epoch in range(1, num_epochs+1):

        for batch in trainloader:

            x,y = batch 
            Yhat = clf(x)

            loss = loss_fn(Yhat, y)

            #backpropogation
            opt.zero_grad()
            loss.backward()
            opt.step()    
            
        if (verbose): print(f"{epoch}/{num_epochs} epochs.")
    
    return clf


def NN_return_eval(model, x, y, n_splits, opt, loss_fn, num_epochs=10, batch_size=128, smote_train = False, verbose = False):

    device = next(model.parameters()).device
    
    X, Y = torch.tensor(x, dtype = torch.float32).to(device), torch.tensor(y, dtype = torch.long).to(device)

    kf = KFold(n_splits=n_splits, shuffle = True, random_state = 18)
    kf.get_n_splits(x)

    overall_metric = 0
    for i, (train_index, test_index) in enumerate(kf.split(x)):

        X_train = X[train_index]
        X_test = X[test_index]

        Y_train = Y[train_index]
        Y_test = Y[test_index]

        if (smote_train):
            smote = SMOTE(random_state=18)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)


        model = train_block(clf = model,
                            opt = opt,
                            loss_fn = loss_fn, 
                            X_train = X_train,
                            Y_train = Y_train, 
                            num_epochs = num_epochs,
                            batch_size = batch_size,
                            verbose = verbose)
        
        
        preds = model(X_test).cpu().detach().numpy()
        
        pred_labels = torch.tensor([x.argmax() for x in preds], dtype = torch.long)
        
        # _______ insert evaluation metric ___________________________________
        
        if (len(set(y)) == 2): average = 'binary'
        else: average = 'weighted'
            
        metric_value = f1_score(Y_test.cpu(), pred_labels.cpu(), average = average)
        
        # ____________________________________________________________________
        
        overall_metric += metric_value
            
    return (overall_metric/n_splits)




def return_best_k_FCNNs(param_dict, 
                        X, Y, 
                        num_top_combinations = 5, 
                        cv_splits = 5, 
                        portion_of_combinations = 0.05, 
                        STC_model_type = 'costco', 
                        rank = 25, 
                        device = 'cpu',
                        verbose = False):

    param_list = list(param_dict)

    tensor_size = [len(param_dict[x]) for x in param_dict]

    total_cells = 1
    for s in tensor_size: total_cells*=s

    num_indices = int(total_cells*portion_of_combinations)

    if (verbose): print(f"{num_indices}/{total_cells} total combinations in sparse tensor.")

    tensor_indices = get_rand_indices(shape = tensor_size, num_indices = num_indices)

    param_combinations = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in tensor_indices]

    values = list()

    it = 0
    for param_combination in param_combinations:
        
        model = FCNN_clf(input_dim = X.shape[1], param_dict = param_combination, num_classes = len(set(Y))).to(device)
        
        if ('batch_size' in param_combination):
            batch_size = param_combination['batch_size']
        else: batch_size = 128

        if ('num_epochs' in param_combination):
            num_epochs = param_combination['num_epochs']
        else: num_epochs = 5

        if ('lr' in param_combination):
            lr = param_combination['lr']
        else: lr = 1e-3
        
        opt = Adam(model.parameters(), lr=lr)                   

        loss_fn = nn.CrossEntropyLoss()

        result = NN_return_eval(model = model, 
                                x = X, 
                                y = Y, 
                                n_splits = cv_splits, 
                                opt = opt, 
                                loss_fn = loss_fn, 
                                num_epochs=num_epochs,                   
                                batch_size=batch_size,                  
                                smote_train=False,
                                verbose = False)     

        values += [result]
        
        it+=1
        if (verbose): print(f"{it}/{len(param_combinations)} param_combinations done.")

    values = torch.tensor(values)

    sparse_tensor = torch.sparse_coo_tensor(indices = torch.tensor(tensor_indices).t(), values= values, size = tensor_size).coalesce().to(device)
        
    if (verbose): print("\nRunning sparse tensor completion...")
    
    STC_model = train_tensor_completion(model_type = STC_model_type, 
                                        train_indices = sparse_tensor.indices().t(),
                                        train_values = sparse_tensor.values(), 
                                        tensor_size = sparse_tensor.size(),
                                        rank = rank, 
                                        num_epochs = 15000, 
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

    # asked perplexity.ai for help with this
    values, indices = torch.topk(tensor.flatten(), num_top_combinations)

    # Convert flat indices back to 3D indices
    top_k_indices = np.array(np.unravel_index(indices.numpy(), tensor.shape)).T
        
        
    best_params = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in top_k_indices]
    
    
    if (verbose): print("\nEvaluating predicted best parameters.")
    
    best_estimated_params = list()

    for i in range(len(best_params)):
        
        parameters = best_params[i]
        
        model = FCNN_clf(input_dim = X.shape[1], param_dict = parameters, num_classes = len(set(Y))).to(device)
        
        if ('batch_size' in param_combination):
            batch_size = param_combination['batch_size']
        else: batch_size = 128

        if ('num_epochs' in param_combination):
            num_epochs = param_combination['num_epochs']
        else: num_epochs = 5

        if ('lr' in param_combination):
            lr = param_combination['lr']
        else: lr = 1e-3
        
        opt = Adam(model.parameters(), lr=lr)                   

        loss_fn = nn.CrossEntropyLoss()

        actual_eval = NN_return_eval(model = model, 
                                     x = X, 
                                     y = Y, 
                                     n_splits = cv_splits, 
                                     opt = opt, 
                                     loss_fn = loss_fn, 
                                     num_epochs=num_epochs,                   
                                     batch_size=batch_size,                  
                                     smote_train=False,
                                     verbose = False)   
        
        predicted_eval = values[i]
                
        best_estimated_params += [(parameters, float(predicted_eval), float(actual_eval))]
        # best_estimated_params += [(parameters, float(actual_eval))]
        
        if (verbose): print(f"{i+1}/{len(best_params)} predicted parameter combinations evaluated.")
        
        
    best_estimated_params.sort(key = lambda x: x[-1], reverse = True)
    
    print("Done!")
    
    return best_estimated_params