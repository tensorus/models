# tensorus-models

This repository provides a collection of machine learning models that extend the core [Tensorus](https://github.com/tensorus/tensorus) project. The models live under the `tensorus.models` namespace and rely on utilities from the main `tensorus` repository (for example `tensorus.tensor_storage`).

## Installation

Install the package in editable mode while in the repository directory:

```bash
pip install -e .
```

When published on PyPI you will also be able to install it with:

```bash
pip install tensorus-models
```

### Example

```python
from tensorus.models import LinearRegressionModel
```

## Running Tests

The tests require the core `tensorus` package so that `tensorus.tensor_storage` can be imported. Either clone the main repository and install it, or install it from PyPI:

```bash
# Option 1: clone Tensorus
git clone https://github.com/tensorus/tensorus
pip install -e tensorus

# Option 2: install from PyPI
pip install tensorus
```

After installing `tensorus` and the requirements for this repository, run:

```bash
pip install -r requirements.txt
pytest
```
