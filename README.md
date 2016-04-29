# CUTEST.py: Python's CUTEst Interface

`CUTEST.py` is a Python interface package for modeling and solving optimization problems from CUTEst collection.

## Dependencies

- [`CUTEst`](https://github.com/optimizers/cutest-mirror)
- [`Numpy`](http://www.numpy.org)
- [`NLP.py`]

## Optional dependencies

- [`PyKrylov`](https://github.com/PythonOptimizers/pykrylov)

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/PythonOptimizers/NLP.py
   ```

2. Check and modify if you need 'site.cfg' in the folder 'cutest.tools'

3. Install:
```bash
    python setup.py install
    python setup.py test
```

## Example

```bash
>>> from cutest.cutestmodel import CUTEstModel

>>> model = CUTEstModel('ROSENBR')

>>> f = model.obj(model.x0)
>>> g = model.grad(model.x0)
>>> H = model.hess(model.x0)
```


