# Demo

Download this folder and use it on its own to tune a model. Install the dependencies below, open a notebook from this directory, and replace the synthetic data, model, and parameter grid with your own. Each notebook scores a sample of that grid and uses sparse tensor completion to fill in the rest. The notebook only sets those inputs and calls into `src/`.

From this folder:

```
pip install -r requirements.txt
```

Open a notebook with this folder as the working directory, so `src` is the copy in this folder.

- `sklearn.ipynb` tunes a scikit-learn model. The example is a random forest classifier, scored with cross-validated F1.
- `mlp.ipynb` tunes a PyTorch network for binary classification, scored with accuracy.

*Please note this* `README.md` *file was created with heavy usage of cursor.ai, apologies for any mistakes.*