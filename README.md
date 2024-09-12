# Sparse Tensor Completion for AutoML

#### Utilizing Sparse Tensor Completion to accelerate Non-Deep Learning Hyperparameter Optimization, Neural Architecture Search, & Database Query Cardinality estimation.


'notebooks' contains the code for tensor generation as well as the code for the expirements performed (and all the tensor completion models themselves).

'classification_datasets' contains the downstream task datasets we used to generate our tensors for this application.

'training_tensors' contains all the tensors we generated for evaluating Sparse Tensor Completion for this application.
  - these are split up into 'non_deep','deep_learning', & 'query_tensors' folders

Naming conventions:
  - non_deep/{model_type}\_{classification\_dataset}\_{date\_created}.pt
  - deep_learning/{model_type}\_{classification\_dataset}\_{date\_created}.pt
  - query_tensors/{conjuctive\_operator\_1}_{conjunctive\_operator\_2}\_{date\_created}.pt


Detailed list of training tensors:
<img width="880" alt="Screenshot 2024-09-12 at 2 17 19 PM" src="https://github.com/user-attachments/assets/bb4d5f9d-e40d-478c-a9b0-5de170f7789c">



See 'demo.ipynb' for usage of this application.
