from notebooks.utilities.helper_functions import *


def return_best_k_params(model, 
                         param_dict, 
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
        
        model.set_params(**param_combination)

        value = return_eval(model = model, x = X, y = Y, n_splits = cv_splits, smote_train=False, random_state = 18)

        values += [value]
        
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

    # if (verbose):
        
    #     print(f"Top {num_top_combinations} values:")
    #     print(values)
    #     print("\nTheir indices (in 3D format):")
    #     print(top_k_indices)

    #     # If you want to pair each value with its 3D index
    #     for value, index in zip(values, top_k_indices):
    #         print(f"Value: {value.item():.4f}, Index: {tuple(index)}")
        
        
    best_params = [{param_list[i]: param_dict[param_list[i]][tensor_index[i]] for i in range(len(tensor_index))} for tensor_index in top_k_indices]
    
    
    if (verbose): print("\nEvaluating predicted best parameters.")
    
    best_estimated_params = list()

    for i in range(len(best_params)):
        
        parameters = best_params[i]
        
        model.set_params(**parameters)
        
        actual_eval = return_eval(model = model, x = X, y = Y, n_splits = cv_splits, smote_train = False, random_state = 18)
        predicted_eval = values[i]
        
        # if (verbose): print(f"{parameters}\nPredicted Evaluation: {predicted_eval:.4f}; Actual Evaluation: {actual_eval:.4f}\n")
        
        # best_estimated_params += [(parameters, float(predicted_eval), float(actual_eval))]
        best_estimated_params += [(parameters, float(actual_eval))]
        
        
    best_estimated_params.sort(key = lambda x: x[-1], reverse = True)
    
    print("Done!")
    
    return best_estimated_params