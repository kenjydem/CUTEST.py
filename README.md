# CUTEST.py: Python's CUTEst Interface

`CUTEST.py` is a Python interface package for modeling and solving optimization problems from CUTEst collection.

## Dependencies

- [`CUTEst`](https://github.com/optimizers/cutest-mirror)
- [`NLP.py`](https://github.com/PythonOptimizers/NLP.py)
- [`Numpy`](http://www.numpy.org)

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/PythonOptimizers/NLP.py
   ```

2. In the folder `cutest/tools`, copy `site.template.cfg` to `site.cfg` and modify to your needs.

3. Install:
```bash
    python setup.py install
    python setup.py test
```

## Example

```bash
>>> from cutest.model.cutestmodel import CUTEstModel

>>> model = CUTEstModel('ROSENBR')

>>> f = model.obj(model.x0)
>>> g = model.grad(model.x0)
>>> H = model.hess(model.x0)
```

## Option

With the sifedecoder, you can check if the problem exists in differents size.
For example, LUBRIFC problem exists in 3 differents size. To import the size you want, you just have to give as argument the sifparameter when you initiate your problem in Python. For example:

```bash
>>> from cutest.model.cutestmodel import CUTEstModel

>>> model = CUTEstModel('LUBRIFC', sifParams=['-param', 'NN=50']) 
```
