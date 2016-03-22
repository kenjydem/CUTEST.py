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
    - x0 : initial estimate of the solution of the problem
    - Lvar : lower bounds on the variables
    - Uvar : upper bounds on the variables
    - pi0 : initial estimate of the Lagrange multipliers
    - Lcon : lower bounds on the inequality constraints
    - Ucon : upper bounds on the inequality constraints
    - equatn : logical array whose i-th component is 1 if the i-th constraint is an equation
    - linear : a logical array whose i-th component is 1 if the i-th constraint is  linear
    - name : name of the problem 
    """

    def __init__(self, name):
        if name[-4:] == ".SIF":
            name = name[:-4]
            
        directory = compile(name)
        os.chdir(directory)
        cc = importlib.import_module(name)
        self.lib = cc.Cutest(name)
        prob = self.lib.loadProb("OUTSDIF.d")
        kwargs = {'x0':prob['x'], 'pi0':prob['v'], 'Lvar':prob['bl'], 'Uvar':prob['bu'], 'Lcon':prob['cl'], 'Ucon':prob['cu']} 
        NLPModel.__init__(self, prob['nvar'], prob['ncon'], name, **kwargs)
        self.nnzj = prob['nnzj']
        self.nnzh = prob['nnzh']
        #self._nlin = len(prob['lin']) In the newest NLPy, it's already compiled
        

    def obj(self,x, **kwargs):
        """ 
        Compute  objective function and constraints at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            f = self.lib.cutest_cfn(self.n, self.m, x, 0)
        else:
            f = self.lib.cutest_ufn(self.n, self.m, x)
        return f

    def grad(self, x, **kwargs):
        """
        Compute objective gradient at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            g = self.lib.cutest_cofg(self.n, self.m, x)
        else:
            g = self.lib.cutest_ugr(self.n, self.m, x)
            return g

    def cons(self, x, **kwargs):
        """
        Evaluate vector of constraints at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        elif i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        else :
            return self.lib.cutest_cfn(self.n, self.m, i, x, 1)

    def icons(self, i, x, **kwargs):
        """
        Evaluate i-th constraint at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        elif i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        else :
            return self.lib.cutest_ccifg(self.n, self.m, i, x, 0)
            
        
    def igrad(self, i, x, **kwargs):
        """
         Evalutate i-th constraint gradient at x
         Gradient is returned as a dense vector
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        elif i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        else :
            return self.lib.cutest_ccifg(self.n, self.m, i, x, 1)


    # Evaluate i-th constraint gradient at x
    # Gradient is returned as a sparse vector
    def sigrad(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate constraints Jacobian at x
    def jac(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate Lagrangian Hessian at (x,z)
    def hess(self, x, z=None, **kwargs):
        """
         Evaluate Lagrangian Hessian at (x,z) if the problem is
         constrained and the Hessian of the objective function if the problem is unconstrained.
        - x: Evaluated point (numpy array)
        """
        if self.m > 0 :
            if z==None :
                raise ValueError('the Lagrange multipliers need to be specified')
            else :
                res = self.lib.cutest_cdh(self.n, self.m, x, z)
        else :
            res = self.lib.cutest_udh(self.n, x)
        return res

    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, z, p, **kwargs):
        """ 
        Evaluate matrix-vector product between the Hessian of the Lagrangian and a vector
        - x: Evaluated point (numpy array)
        - z: The Lagrangian (numpy array)
        - p: A vector (numpy array)
        """
        if self.m > 0 :
            if z==None :
                raise ValueError('the Lagrange multipliers need to be specified')
            res = self.lib.cutest_chprod(self.n, self.m, x, z, p)
        else :
            res = self.lib.cutest_hprod(self.n, x, p)
        return res

    # Evaluate matrix-vector product between
    # the Hessian of the i-th constraint and a vector
    def hiprod(self, i, x, p, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    def unewton(self):
        """ solve the problem using Newton's algorithm """
        if self.m  > 0 :
            raise TypeError('This method only works on unconstrained problems')
            
        x = self.x0
        gx = self.grad(x)
        Hx = self.hess(x)
        k = 0
        while(np.linalg.norm(gx) > 1.0e-6):
            print "%2d  %7.1e" % (k, np.linalg.norm(gx))
            x -= np.linalg.solve(Hx, gx)
            gx = self.grad(x)
            Hx = self.hess(x)
            k = k + 1
        return x
