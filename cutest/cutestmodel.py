#*-coding:Utf-8 -*
import os, importlib, sys, subprocess, numpy as np
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
        cur_dir = os.getcwd() 
        directory = compile(name)
        os.chdir(directory)
        cc = importlib.import_module(name) 
        self.lib = cc.Cutest(name, "OUTSDIF.d")
        kwargs = {'x0':self.lib.x, 'pi0':self.lib.v, 'Lvar':self.lib.bl, 'Uvar':self.lib.bu, 'Lcon':self.lib.cl, 'Ucon':self.lib.cu} 
        NLPModel.__init__(self, self.lib.nvar, self.lib.ncon, name, **kwargs)
        self.nnzj = self.lib.nnzj
        self.nnzh = self.lib.nnzh
        self.directory = directory
        #self._nlin = len(self.lib.lin) In the newest NLPy, it's already compiled
        os.chdir(cur_dir)

    def obj(self,x, **kwargs):
        """ 
        Compute  objective function and constraints at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            f = self.lib.cutest_cfn(x, 0)
        else:
            f = self.lib.cutest_ufn(x)
        return f

    def grad(self, x, **kwargs):
        """
        Compute objective gradient at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            g = self.lib.cutest_cofg(x)
        else:
            g = self.lib.cutest_ugr(x)
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
            return self.lib.cutest_cfn(i, x, 1)

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
            return self.lib.cutest_ccifg(i, x, 0)
            
        
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
            return self.lib.cutest_ccifg(i, x, 1)


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
                res = self.lib.cutest_cdh(x, z)
        else :
            res = self.lib.cutest_udh(x)
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
            res = self.lib.cutest_chprod(x, z, p)
        else :
            res = self.lib.cutest_hprod(x, p)
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

    def __del__(self):
        """Delete problem"""
        sys.modules[self.name] = None
        cmd = ['rm']+['-rf']+[self.directory]
        subprocess.call(cmd)
        
