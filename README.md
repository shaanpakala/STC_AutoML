# Automating Data Science Pipelines with Tensor Completion

Paper: [[Link](https://ieeexplore.ieee.org/document/10825934)] [[PDF](https://arxiv.org/pdf/2410.06408)]

Contact: [shaan.pakala@gmail.com](mailto:shaan.pakala@gmail.com)

## Usage

To tune your own model, download `[demo/](demo/)` and use that folder on its own (it is entirely self-contained). Install from `[demo/requirements.txt](demo/requirements.txt)`, then run a notebook with `demo/` as the working directory. The notebooks ship with a small synthetic example. Swap in your data, model, and parameter grid. Details are in `[demo/README.md](demo/README.md)`.

- `[demo/sklearn.ipynb](demo/sklearn.ipynb)` — scikit-learn models
- `[demo/mlp.ipynb](demo/mlp.ipynb)` — a PyTorch neural network



## Data and experiments

- `notebooks/` — tensor generation and the experiments from the paper
- `classification_datasets/` — downstream datasets used to build the training tensors
- `training_tensors/` — tensors generated for the sparse tensor completion experiments. Details are in `[training_tensors/README.md](training_tensors/README.md)`.



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

