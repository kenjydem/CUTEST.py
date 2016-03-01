#*-coding:Utf-8 -*

import sys
import numpy as np
from nlp.model.nlpmodel import NLPModel
from cutest.ccutest import cutest_cfn,cutest_ufn

class CUTEstModel(NLPModel) :
    """Classe définissant un problème :
    - n : number of variables
    - m : number of constraints
    - nnzj : number of nonzeros constraint in Jacobian
    - nnzh : number of nonzeros in Hessian of Lagrangian    
    - x : initial estimate of the solution of the problem
    - bl : lower bounds on the variables
    - bu : upper bounds on the variables
    - v : initial estimate of the Lagrange multipliers
    - cl : lower bounds on the inequality constraints
    - cu : upper bounds on the inequality constraints
    - equatn : logical array whose i-th component is 1 if the i-th constraint is an equation
    - linear : a logical array whose i-th component is 1 if the i-th constraint is  linear
    - name : name of the problem 
    """

    def __init__(self, n, m, nnzj, nnzh, x, bl, bu, v,
                  cl, cu, equatn, linear, name) :
        kwargs = {'x0':x, 'pi0':v, 'Lvar':bl, 'Uvar':bu, 'Lcon':cl, 'Ucon':cu} 
        NLPModel.__init__(self, n, m, name, **kwargs)
        self.nnzj = nnzj
        self.nnzh = nnzh
        self.equatn = equatn
        self._lin = linear
        self._nlin = len(self.lin)
        self.x = x 
        self.c = np.zeros((m,), dtype=np.double)
        self.f = 0
        self.status = 0
     
    # Evaluate objective function at x
    def obj(self,x, **kwargs):
        """ Evalue la fonction objective et les contraintes du problème:
        - x: Evaluation point (numpy array)
        """
        if self.m > 0:
             [self.c, self.f] = cutest_cfn(self.status, self.n, self.m, x, self.f, self.c)
        else:
            self.f = cutest_ufn(self.status, self.n, x, self.f)

    # Evaluate objective gradient at x
    def grad(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate vector of constraints at x
    def cons(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate i-th constraint at x
    def icons(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evalutate i-th constraint gradient at x
    # Gradient is returned as a dense vector
    def igrad(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate i-th constraint gradient at x
    # Gradient is returned as a sparse vector
    def sigrad(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate constraints Jacobian at x
    def jac(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate Lagrangian Hessian at (x,z)
    def hess(self, x, z=None, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, z, p, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate matrix-vector product between
    # the Hessian of the i-th constraint and a vector
    def hiprod(self, i, x, p, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'
