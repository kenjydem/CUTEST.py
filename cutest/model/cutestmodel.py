#*-coding:Utf-8 -*
import os, sys, importlib, subprocess, numpy as np
from nlp.model.nlpmodel import NLPModel
import scipy.sparse as sparse
from cutest.tools.compile import compile_SIF

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
        
        sys.path[0:0] = ['.']  # prefix current directory to list
        directory = compile_SIF(name)
        cur_dir = os.getcwd()
        os.chdir(directory)
        cc = importlib.import_module(name)
        self.lib = cc.Cutest(name)
        prob = self.lib.loadProb("OUTSDIF.d")
        kwargs = {'x0':prob['x'], 'pi0':prob['v'], 'Lvar':prob['bl'], 'Uvar':prob['bu'], 'Lcon':prob['cl'], 'Ucon':prob['cu']} 
        NLPModel.__init__(self, prob['nvar'], prob['ncon'], name, **kwargs)
        self.nnzj = prob['nnzj']
        self.nnzh = prob['nnzh']
        self.directory = directory
        os.chdir(cur_dir)
        #self._nlin = len(prob['lin']) In the newest NLPy, it's already compiled
        

    def obj(self,x, **kwargs):
        """ 
        Compute  objective function and constraints at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            f = self.lib.cutest_cfn(self.n, self.m, x, 0)
            return f    
        
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

    def sgrad(self, x) : 
        """
        Compute objective gradient at x:
        Gradient is returned as a sparse vector
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            print 'the problem is unconstrained : the dense gradient is return instead'
            return  self.grad(x)

        g, ir = self.lib.cutest_cofsg(self.n, x)
        ir[ir.nonzero()] = ir[ir.nonzero()] - 1
        return sparse.coo_matrix((g, (np.zeros((self.n,), dtype=int), ir)), shape=(1,self.n))

    def cons(self, x, **kwargs):
        """
        Evaluate vector of constraints at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        return self.lib.cutest_cfn(self.n, self.m, x, 1)

    def icons(self, i, x, **kwargs):
        """
        Evaluate i-th constraint at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        return self.lib.cutest_ccifg(self.n, self.m, i, x, 0)
            
        
    def igrad(self, i, x, **kwargs):
        """
         Evalutate i-th constraint gradient at x
         Gradient is returned as a dense vector
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        return self.lib.cutest_ccifg(self.n, self.m, i, x, 1)


    # Evaluate i-th constraint gradient at x
    # Gradient is returned as a sparse vector
    def sigrad(self, i, x, **kwargs):
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))

        g, ir = self.lib.cutest_ccifsg(self.n, self.nnzj, i, x)
        ir[ir.nonzero()] = ir[ir.nonzero()] - 1
        return sparse.coo_matrix((g, (np.zeros((self.n,), dtype=int), ir)), shape=(1,self.n))
            
    def jac(self, x, **kwargs):
        """  Evaluate constraints Jacobian at x """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        return self.lib.cutest_ccfg(self.n, self.m, x)
            
    def hess(self, x, z=None, **kwargs):
        """
         Evaluate Lagrangian Hessian at (x,z) if the problem is
         constrained and the Hessian of the objective function if the problem is unconstrained.
        - x: Evaluated point (numpy array)
        """
        if self.m > 0 :
            if z is None :
                raise ValueError('the Lagrange multipliers need to be specified')
            res = self.lib.cutest_cdh(self.n, self.m, x, z)
        else :
            res = self.lib.cutest_udh(self.n, x)
        return res

    def shess(self, x, z=None) :
        """
        Evaluate the Hessian matrix of the Lagrangian, or of the objective if the problem
        is unconstrained, in sparse format
        """
        if self.m > 0:
            if z is None:
                raise ValueError('the Lagrange multipliers need to be specified')
            h, irow, jcol = self.lib.cutest_csh(self.n, self.m, self.nnzh, x, z)
        else:
            h, irow, jcol = self.lib.cutest_ush(self.n, self.nnzh, x)
        
        #on ne récupère que la partie triangulaire supérieure de h
        # nous reconstruisons donc la matrice h       
        offdiag = 0
        for i in range(self.nnzh):
            if irow[i] != jcol[i]:
                k = self.nnzh + offdiag
                irow[k] = jcol[i];
                jcol[k] = irow[i];
                h[k] = h[i];
            offdiag += 1;
        
        h = h[irow.nonzero()]
        irow = irow[irow.nonzero()] - 1
        jcol = jcol[jcol.nonzero()] - 1

        return sparse.coo_matrix((h, (irow, jcol)), shape=(self.n,self.n))

    def ihess(self, x, i) : 
        """
        Return the dense Hessian of the objective or of a constraint. The
        function index is ignored if the problem is unconstrained.
        Usage:  Hi = ihess(x, i)
        """
        if self.m == 0 :
            print 'Warning : the problem is unconstrained, the dense Hessian of the objective is returned'
            return self.lib.cutest_udh(self.n, x)
        if i > self.m :
                raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        return self.lib.cutest_cidh(self.n, x, i)
        
    def ishess(self, x, i) :
        """
        Evaluate the Hessian matrix of the i-th problem function (i=0 is the objective
function), or of the objective if problem is unconstrained, in sparse format
        """
        if self.m == 0:
             return self.shess(x,i)
        if i > self.m :
                raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        h, irow, jcol = self.lib.cutest_cish(self.n, self.nnzh, x, i)     
        offdiag = 0
        for i in range(self.nnzh):
            if irow[i] != jcol[i]:
                k = self.nnzh + offdiag
                irow[k] = jcol[i];
                jcol[k] = irow[i];
                h[k] = h[i];
            offdiag += 1;
        
        h = h[irow.nonzero()]
        irow = irow[irow.nonzero()] - 1
        jcol = jcol[jcol.nonzero()] - 1

        return sparse.coo_matrix((h, (irow, jcol)), shape=(self.n,self.n))
           

    def hprod(self, x, z, p, **kwargs):
        """ 
        Evaluate matrix-vector product between the Hessian of the Lagrangian and a vector
        - x: Evaluated point (numpy array)
        - z: The Lagrangian (numpy array)
        - p: A vector (numpy array)
        """
        if self.m > 0 :
            if z is None :
                raise ValueError('the Lagrange multipliers need to be specified')
            res = self.lib.cutest_chprod(self.n, self.m, x, z, p)
        else :
            res = self.lib.cutest_hprod(self.n, x, p)
        return res

    
    def hiprod(self, i, x, p, **kwargs):
        """
        Evaluate matrix-vector product between
        the Hessian of the i-th constraint and a vector
        """
        if self.m == 0 :
            raise TypeError('the problem ' + self.name + ' does not have constraints')
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        return np.dot(self.lib.cutest_cidh(self.n, x, i), p)

    def jprod(self, x, z) :
        """
        Evaluate the matrix-vector product between the Jacobian and a vector
        """
        dim = z.shape
        if len(dim) != 1 or dim[0] != self.n :
            raise ValueError('the vector z dimension should be ['+str(self.n) +' 1]')
        if self.m == 0 :
            raise ValueError('this function is only available for constrained problems')
        return self.lib.cutest_cjprod(self.n, self.m, x, z, 0)
    
    def jtprod(self, x, z) :
        """
        Evaluate the matrix-vector product between the transpose Jacobian and a vector
        """
        dim = z.shape
        if len(dim) != 1 or dim[0] != self.m :
            raise ValueError('the vector z dimension should be ['+str(self.m)+' 1]')
        if self.m == 0 :
            raise ValueError('this function is only available for constrained problems')
        return self.lib.cutest_cjprod(self.n, self.m, x, z, 1)

    def __del__(self):
        """
        Delete problem
        """
        del(sys.modules[self.name])
        cmd = ['rm']+['-rf']+[self.directory]
        subprocess.call(cmd)
