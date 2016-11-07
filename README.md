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

2. Check and modify if you need 'site.cfg' in the folder 'cutest.tools'

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

 With the sifedecoder, you can check if the problem exist in different size.
For example, LUBRIFC problem exist in 3 different size. To import the size you want, you just have to give sif parameter as argument when you import your problem in Python. For example:

```bash
>>> from cutest.model.cutestmodel import CUTEstModel

>>> model = CUTEstModel('LUBRIFC', sifParams=['-param', 'NN=10']) 
```
