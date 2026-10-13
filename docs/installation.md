# Installation

## Requirements

- Python >= 3.10, < 3.14

## Install from GitHub

```bash
pip install git+https://github.com/BellaicheLab/vgep.git
```

## Optional: PERLER

PERLER is not on PyPI, and it pins `scipy==1.6.0`, which fails to build on
Python 3.10+. Install it from GitHub with `--no-deps`:

```bash
pip install "git+https://github.com/yasokochi/Perler.git" --no-deps
```

Then edit `perler/perler_class.py` line 348 to
`DM = distance.cdist(self.__mu, self.__r, metric="mahalanobis", VI=VI)`;
perler passes that matrix as `V=`, which every SciPy >= 1.7 rejects.

## Optional: Tangram

Tangram runs on PyTorch (install `torch` first); its hyperparameter search uses
[Optuna](https://optuna.org):

```bash
pip install tangram-sc optuna
```

CytoSPACE needs no extra package: its assignment problem is solved with
`scipy.optimize.linear_sum_assignment`.
