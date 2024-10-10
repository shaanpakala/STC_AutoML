# Automating Data Science Pipelines with Tensor Completion

Paper: https://arxiv.org/abs/2410.06408

Authors: Shaan Pakala, Bryce Graw, Dawon Ahn, Tam Dinh, Mehnaz Tabassum Mahin, Vassilis Tsotras, Jia Chen, and Evangelos Papalexakis

#### Utilizing Sparse Tensor Completion to accelerate Non-Deep Learning Hyperparameter Optimization, Neural Architecture Search, & Database Query Cardinality estimation.

##### See demo.ipynb for usage of this application.

'notebooks' contains the code for tensor generation as well as the code for the experiments performed (and all the tensor completion models themselves).

'classification_datasets' contains the downstream task datasets we used to generate our tensors for this application.

'training_tensors' contains all the tensors we generated for evaluating Sparse Tensor Completion for this application.
  - these are split up into 'non_deep','deep_learning', & 'query_tensors' folders

Naming conventions:
  - non_deep/{model_type}\_{classification\_dataset}\_{date\_created}.pt
  - deep_learning/{model_type}\_{classification\_dataset}\_{date\_created}.pt
  - query_tensors/{conjuctive\_operator\_1}_{conjunctive\_operator\_2}\_{date\_created}.pt


Detailed list of training tensors:

<img width="875" alt="Screenshot 2024-09-12 at 2 18 12 PM" src="https://github.com/user-attachments/assets/2b95f2cb-f21f-406d-a7cd-c4a6c26566a0">



If you use our code or datasets, please cite:

```
@misc{pakala2024automatingdatasciencepipelines,
      title={Automating Data Science Pipelines with Tensor Completion}, 
      author={Shaan Pakala and Bryce Graw and Dawon Ahn and Tam Dinh and Mehnaz Tabassum Mahin
              and Vassilis Tsotras and Jia Chen and Evangelos E. Papalexakis},
      year={2024},
      eprint={2410.06408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06408}, 
}
```
