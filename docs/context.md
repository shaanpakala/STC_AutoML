# Agent context

Finished: Tuesday, October 6, 2026, 6:38 PM PDT

**Demo snapshot:** commit `Add a self-contained demo for tuning models.` on `main`

- Hash: `cc15a6a079cec8a6a1c9b1dc1861da79ea0fceb0`
- Commit: https://github.com/shaanpakala/STC_AutoML/commit/cc15a6a079cec8a6a1c9b1dc1861da79ea0fceb0
- Repo: https://github.com/shaanpakala/STC_AutoML

That commit adds `demo/` and the README that points users there. Restore it to get this tuning folder back.

**Earlier snapshot:** commit `original code.` on `main`

- Hash: `bf481d759f0edb99b485b3c32c9e908e0cfeea0f`
- Commit: https://github.com/shaanpakala/STC_AutoML/commit/bf481d759f0edb99b485b3c32c9e908e0cfeea0f

That commit is the published tree from before `demo/` was added. This file was updated after the demo push, so the demo commit does not contain this revision of the note.

Read this file together with `docs/paper.pdf` before changing anything.

## Published paper: do not edit the experimental code

The paper is already published. None of the experimental code is allowed to be touched or edited, ever.

Do not edit anything under:

- `classification_datasets/`
- `notebooks/`
- `src/`
- `training_tensors/`

Those directories hold the published experiments, the downstream datasets, the generated meta-datasets, and the method implementations that match the paper. Read them. Leave them unchanged. New practical tuning code belongs in `demo/`, not in those four directories.

## The paper

`docs/paper.pdf` is *Automating Data Science Pipelines with Tensor Completion* (Pakala et al., 2024 IEEE International Conference on Big Data, pp. 1075–1084, DOI `10.1109/BigData62323.2024.10825934`). IEEE Xplore: https://ieeexplore.ieee.org/document/10825934. arXiv: https://arxiv.org/pdf/2410.06408. Citation is in `README.md`.

The paper treats combinatorial data-science design problems as sparse tensor completion. Each search variable is one mode of a tensor, and each entry is the outcome of one configuration. A small set of observed entries is used to estimate the rest of the grid. The tasks are hyperparameter optimization for non-neural models, neural architecture search, and query cardinality estimation (including distinct cardinality). The methods include existing tensor completion models plus a smoothness-constrained CPD variant (CPD-S) and ensembles of those models. Experiments, dataset construction, and the generated tensors are in the paper, especially the dataset-generation section, Table 1, and the experimental evaluation. Use `docs/paper.pdf` when those details matter. This note does not replace it.

## Practical tuning: `demo/`

`demo/` is the folder a user should download and use on its own to tune a model. It stays inside this repository. It was kept here rather than split into a second repo. The notebooks ship with a synthetic example. The user replaces the data, model, and parameter grid.

Run the notebooks with `demo/` as the working directory, so `import src...` loads `demo/src/` and not the frozen `src/` at the repository root.

`demo/` contains:

- `demo/README.md` — how to install and run the folder by itself.
- `demo/requirements.txt` — numpy, pandas, scikit-learn, torch, ipykernel.
- `demo/sklearn.ipynb` — scikit-learn tuning. It calls `return_best_k_params` in `demo/src/sklearn_grid_search.py`. The example is a random forest classifier scored with cross-validated F1. The sample size argument is `tensor_portion`: a fraction when it is below 1, and a count of grid cells when it is 1 or greater.
- `demo/mlp.ipynb` — PyTorch network tuning for binary classification, scored with accuracy. It calls `return_best_k_params` in `demo/src/MLP_grid_search.py`.
- `demo/src/sklearn_grid_search.py` and `demo/src/MLP_grid_search.py` — the only code those notebooks need. There is no `utilities/` subdirectory.

Both demos complete the grid with smooth CP decomposition (`cpd.smooth`). List entries in the parameter dictionary are searched. A single value is held fixed.

MLP-specific details:

- `hidden_dims` lists hidden-layer widths only. The code appends an output layer. `[64]` is a 2-layer network. `[64, 32, 16]` is a 4-layer network.
- `tensor_entries` is how many grid cells to train. `tensor_portion` is the fraction used when `tensor_entries` is not set. If `tensor_entries` is set, it replaces `tensor_portion`.
- `num_tests` repeated evaluations walk forward through a shuffled index list in windows of `training_values`. When the next window would run past the end of the dataset, `return_nn_eval` reshuffles and takes a fresh window, so a setting such as 3 tests of 500 rows on 1,000 samples does not train on an empty set.

Root notebooks `demo_sklearn.ipynb`, `demo_nn.ipynb`, and `demo.ipynb` are still in the repository. `README.md` directs practical use to `demo/`.

## Repository organization

Top-level contents:

- `README.md` — paper links, contact, pointer to `demo/`, data locations, and the citation.
- `demo/` — self-contained tuning folder described above.
- `demo_sklearn.ipynb`, `demo_nn.ipynb`, `demo.ipynb` — earlier root notebooks. Practical use is `demo/`.
- `classification_datasets/` — downstream task datasets used to generate training tensors. Frozen.
- `notebooks/` — code that builds the tensors and runs the paper’s experiments. Frozen. Subdirectories: `notebooks/dataset_creation/` and `notebooks/experiments/`.
- `src/` — published tensor completion implementations and grid-search utilities. Frozen. Subdirectories: `src/tensor_completion_models/` and `src/utilities/`.
- `training_tensors/` — the meta-datasets generated for the paper (PyTorch `.pt` tensors). Frozen. See `training_tensors/README.md`. Subdirectories: `non_deep/` (sklearn-style hyperparameter tensors, plus `non_deep/dataset_mode/` for the cross-dataset setting), `deep_learning/` (neural architecture search tensors), and `query_tensors/` (query cardinality and distinct-cardinality tensors).
- `docs/` — `paper.pdf` and this file (`docs/context.md`).
