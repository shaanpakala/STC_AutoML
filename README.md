# Automating Data Science Pipelines with Tensor Completion

Paper: [[Link](https://ieeexplore.ieee.org/document/10825934)] [[PDF](https://arxiv.org/pdf/2410.06408)]

Contact: shaan.pakala@gmail.com

## Usage

##### To use tensor completion for hyperparameter tuning, download src/ folder and see one of the demo .ipynb notebooks

• `See demo_sklearn.ipynb for hyperparameter tuning sklearn ML models`

• `See demo_nn.ipynb for hyperparameter tuning a PyTorch neural network for regression or classification tasks`

## Data & Experimental Details

'notebooks' contains the code for tensor generation as well as the code for the experiments performed.

'classification_datasets' contains the downstream task datasets we used to generate our training tensors for our experiments.

'training_tensors' contains all the tensors we generated for evaluating Sparse Tensor Completion for our experiments.
  - see 'training_tensors/README.md' for more details


## Citation:

```
@inproceedings{pakala2024automating,
  title={Automating Data Science Pipelines with Tensor Completion},
  author={Pakala, Shaan and Graw, Bryce and Ahn, Dawon and Dinh, Tam and Mahin, Mehnaz Tabassum and Tsotras, Vassilis and Chen, Jia and Papalexakis, Evangelos E},
  booktitle={2024 IEEE International Conference on Big Data (BigData)},
  pages={1075--1084},
  year={2024},
  organization={IEEE}
}
```
