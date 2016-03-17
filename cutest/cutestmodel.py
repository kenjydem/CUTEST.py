#*-coding:Utf-8 -*
import os
import importlib
import sys
import numpy as np
from nlp.model.nlpmodel import NLPModel
from tools import compile

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

    def __init__(self, name):
        if name[-4:] == ".SIF":
            name = name[:-4]
        
        directory = compile(name)
        fname = os.path.join(directory, "OUT.d") #"/OUTSDIF.d"
        os.chdir(directory)
        cc = importlib.import_module(name) 
        self.prob = cc.Cutest(name, fname)
        kwargs = {'x0':self.prob.x, 'pi0':self.prob.v, 'Lvar':self.prob.bl, 'Uvar':self.prob.bu, 'Lcon':self.prob.cl, 'Ucon':self.prob.cu} 
        NLPModel.__init__(self, self.prob.nvar, self.prob.ncon, name, **kwargs)
        self.nnzj = self.prob.nnzj
        self.nnzh = self.prob.nnzh
        self._nlin = len(self.prob.lin)
        self.x = self.prob.x 
        self.c = np.zeros((self.prob.ncon,), dtype=np.double)
        self.f = 0
        self.get_grad = 1 #Check if gradient exist that should be automatic
        self.g = np.zeros((self.prob.nvar,),dtype=np.double)
        self.status = 0

    def obj(self,x, **kwargs):
        """ 
        Compute  objective function and constraints at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            [c, f] = self.prob.cutest_cfn(x, self.f, self.c)
            return c,f    
        else:
            f = self.prob.cutest_ufn(x, self.f)
            return f

    def grad(self, x, **kwargs):
        """
        Compute objective gradient at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            [f, g] = self.prob.cutest_cofg(x, self.f, self.g, self.get_grad)
            return f,g

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
