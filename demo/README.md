# demo

Download this folder and use it on its own to tune a model. Install the dependencies below, open a notebook from this directory, and replace the synthetic data, model, and parameter grid with your own. Each notebook scores a sample of that grid and uses sparse tensor completion to fill in the rest. The notebook only sets those inputs and calls into `src/`.

From this folder:

```
pip install -r requirements.txt
```

Open a notebook with this folder as the working directory, so `src` is the copy in this folder.

- [`sklearn.ipynb`](sklearn.ipynb) tunes a scikit-learn model. The example is a random forest classifier, scored with cross-validated F1. The call is `return_best_k_params` in `src/sklearn_grid_search.py`.
- [`mlp.ipynb`](mlp.ipynb) tunes a PyTorch network for binary classification, scored with accuracy. `hidden_dims` are hidden-layer widths, and an output layer is added after them, so `[64]` is a 2-layer network and `[64, 32, 16]` is a 4-layer network. The call is `return_best_k_params` in `src/MLP_grid_search.py`.

Both demos use smooth CP decomposition (`cpd.smooth`) to complete the grid. List entries in the parameter dictionary are searched. A single value is held fixed. `tensor_entries` is how many grid cells to train. `tensor_portion` is the fraction to train when a count is not given.
