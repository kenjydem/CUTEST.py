import os, sys, importlib, subprocess, numpy as np
from nlp.model.nlpmodel import NLPModel
from nlp.model.qnmodel import QuasiNewtonModel
import scipy.sparse as sparse
from cutest.tools.compile import compile_SIF

class CUTEstModel(NLPModel) :
    """A general class from NLP.py :
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

    def __init__(self, name, **kwargs):
        if name[-4:] == ".SIF":
            name = name[:-4]
        
        sys.path[0:0] = ['.']  # prefix current directory to list
        sifParams = kwargs.get("sifParams",None) 
        directory, cython_lib_name = compile_SIF(name,sifParams)
        cur_dir = os.getcwd()
        os.chdir(directory)
        cc = importlib.import_module(cython_lib_name)
        self.lib = cc.Cutest(name)
        
        prob = self.lib.loadProb("OUTSDIF.d")
        
        NLPModel.__init__(self, prob['nvar'], prob['ncon'], name,
                          x0 = prob['x'], pi0 = prob['v'], 
                          Lvar = prob['bl'], Uvar = prob['bu'],
                          Lcon = prob['cl'], Ucon = prob['cu'])
        
        self.nnzj = prob['nnzj']
        self.nnzh = prob['nnzh']
        self.directory = directory
        os.chdir(cur_dir)
        

    def obj(self,x, **kwargs):
        """ 
        Compute objective function and constraints at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            f = self.lib.cutest_cfn(self.n, self.m, x, 0)
        else:
            f = self.lib.cutest_ufn(self.n, x)

        if self.scale_obj:
            f *= self.scale_obj
        return f
    
    def grad(self, x):
        """
        Compute objective gradient at x:
        - x: Evaluated point (numpy array)
        """
        if self.m > 0:
            f, g = self.lib.cutest_cofg(self.n, self.m, x)
        else:
            g = self.lib.cutest_ugr(self.n, x)
        if self.scale_obj:
            g *= self.scale_obj
        return g

    def sgrad(self, x) : 
        """
        Compute objective gradient at x:
        Gradient is returned as a sparse vector
        - x: Evaluated point (numpy array)
        """

        if self.m == 0 :
            print 'Function unimplemented for unconstriant problem'
        else:
            f, (vals, rows) = self.lib.cutest_cofsg(self.n, x)
        
        if self.scale_obj:
            vals *= self.scale_obj
        return (vals, rows)

    def cons(self, x, **kwargs):
        """
        Evaluate vector of constraints at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            return np.array([], dtype=np.double)
        else:
            f, c = self.lib.cutest_cfn(self.n, self.m, x, 1)

            if isinstance(self.scale_con, np.ndarray):
                c *= self.scale_con
            return c

    def icons(self, i, x, **kwargs):
        """
        Evaluate i-th constraint at x
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            return np.array([], dtype=np.double)
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        ci = self.lib.cutest_ccifg(self.n, self.m, i, x, 0)
        if isinstance(self.scale_con, np.ndarray):
            ci *= self.scale_con[i]
        return ci
            
    def igrad(self, i, x, **kwargs):
        """
         Evalutate i-th constraint gradient at x
         Gradient is returned as a dense vector
        - x: Evaluated point (numpy array)
        """
        if self.m == 0 :
            return np.array((0,self.n), dtype=np.double)
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        gi = self.lib.cutest_ccifg(self.n, self.m, i, x, 1)
        if isinstance(self.scale_con, np.ndarray):
            gi *= self.scale_con[i]
        return gi

    def sigrad(self, i, x):
        """
        Evaluate i-th constraint gradient at x
        Gradient is returned as a sparse vector
        """
        if self.m == 0 :
            return sparse.coo_matrix((0,self.n),dtype=np.double)
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        if i==0:
            raise ValueError('i must be between 1 and '+str(self.m))

        g, ir = self.lib.cutest_ccifsg(self.n, self.nnzj, i, x)
        ir[ir.nonzero()] = ir[ir.nonzero()] - 1
        sci = sparse.coo_matrix((g, (np.zeros((self.n,), dtype=int), ir)), shape=(1,self.n))
        if isinstance(self.scale_con, np.ndarray):
            sci *= self.scale_con[i]
        return sci
            
    def jac(self, x):
        """  Evaluate constraints Jacobian at x """
        if self.m == 0 :
            return np.array((0,self.n), dtype=np.double)
        J = self.lib.cutest_ccfg(self.n, self.m, x)

        if isinstance(self.scale_con, np.ndarray):
            J = (self.scale_con * J.T).T

        return J

    def sjac(self, x, z=None):

        """ 
        Evaluate Jacobian in a sparse format 

        - x: Evaluated point (numpy array)
        - z: Lagrange multipliers
        """ 

        if self.m > 0:
            if z is None:
                raise ValueError('the Lagrange multipliers need to be specified')
            if isinstance(self.scale_con, np.ndarray):
                z = z.copy()
                z *= self.scale_con
                if self.scale_obj:
                    z /= self.scale_obj       

            rows, cols, vals = self.lib.cutest_csgr(self.n, self.m, self.nnzj, x, -z)
                            
        if self.scale_obj:
            vals *= self.scale_obj
        
        return (rows, cols, vals)


    def hess_dense(self, x, z=None):
        """
         Evaluate Lagrangian Hessian at (x,z) if the problem is
         constrained and the Hessian of the objective function if the problem is unconstrained.
        - x: Evaluated point (numpy array)
        - z: Lagrange multipliers

        NOTE: CUTEst Lagrangian is L(x,z) = f(x) + z'c(x), so we pass -z to compensate
        """
        if self.m > 0 :
            if z is None :
                raise ValueError('the Lagrange multipliers need to be specified')
            if isinstance(self.scale_con, np.ndarray):
                z = z.copy()
                z *= self.scale_con
                if self.scale_obj:
                    z /= self.scale_obj
            hes = self.lib.cutest_cdh(self.n, self.m, x, -z)
        else :
            hes = self.lib.cutest_udh(self.n, x)

        if self.scale_obj:
            hes *= self.scale_obj
        return hes

    def hess(self, x, z=None) :
        """
        Evaluate the Hessian matrix of the Lagrangian, or of the objective if the problem
        is unconstrained, in sparse format

        NOTE: CUTEst Lagrangian is L(x,z) = f(x) + z'c(x), so we pass -z to compensate
        """
        if self.m > 0:
            if z is None:
                raise ValueError('the Lagrange multipliers need to be specified')
            if isinstance(self.scale_con, np.ndarray):
                z = z.copy()
                z *= self.scale_con
                if self.scale_obj:
                    z /= self.scale_obj

            rows, cols, vals = self.lib.cutest_csh(self.n, self.m, self.nnzh, x, -z)
        else:
            rows, cols, vals = self.lib.cutest_ush(self.n, self.nnzh, x)
        
        if self.scale_obj:
            h *= self.scale_obj

        return (rows, cols, vals)

    def ihess(self, x, i) : 
        """
        Return the dense Hessian of the objective or of a constraint. The
        function index is ignored if the problem is unconstrained.
        Usage:  Hi = ihess(x, i)
        """
        if self.m == 0 :
            print 'Warning : the problem is unconstrained, the dense Hessian of the objective is returned'
            h = self.lib.cutest_udh(self.n, x)
            if self.scale_obj:
                h *= self.scale_obj
            return h
        if i > self.m :
            raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        h = self.lib.cutest_cidh(self.n, x, i)
        if isinstance(self.scale_con, np.ndarray):
            h *= self.scale_con[i]
        return h
        
    def ishess(self, x, i) :
        """
        Evaluate the Hessian matrix of the i-th problem function (i=0 is the objective
        function), or of the objective if problem is unconstrained, in sparse format
        """
        if self.m == 0:
             return self.ihess(x,i)
        if i > self.m :
                raise ValueError('the problem ' + self.name + ' only has ' + str(self.m) + ' constraints')
        h, irow, jcol = self.lib.cutest_cish(self.n, self.nnzh, x, i)     

        if isinstance(self.scale_con, np.ndarray):
            h *= self.scale_con[i]

        # We rebuild the matrix h from the upper triangle       
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
        - z: The Lagrange multipliers (numpy array)
        - p: A vector (numpy array)

        NOTE: CUTEst Lagrangian is L(x,z) = f(x) + z'c(x), so we pass -z to compensate
        """
        if self.m > 0 :
            if z is None :
                raise ValueError('the Lagrange multipliers need to be specified')
            if isinstance(self.scale_con, np.ndarray):
                z = z.copy()
                z *= self.scale_con
                if self.scale_obj:
                    z /= self.scale_obj
            res = self.lib.cutest_chprod(self.n, self.m, x, -z, p)
        else :
            res = self.lib.cutest_hprod(self.n, x, p)
        if self.scale_obj:
            res *= self.scale_obj
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
        res = np.dot(self.lib.cutest_cidh(self.n, x, i), p)
        if isinstance(self.scale_con, np.ndarray):
            res *= self.scale_con[i]
        return res

    def jprod(self, x, p) :
        """
        Evaluate the matrix-vector product between the Jacobian and a vector
        """
        dim = p.shape
        if len(dim) != 1 or dim[0] != self.n :
            raise ValueError('the vector p dimension should be ['+str(self.n) +' 1]')
        if self.m == 0 :
            return np.array([], dtype=np.double)
        prod = self.lib.cutest_cjprod(self.n, self.m, x, p, 0)
        if isinstance(self.scale_con, np.ndarray):
            prod *= self.scale_con
        return prod
    
    def jtprod(self, x, p) :
        """
        Evaluate the matrix-vector product between the transpose Jacobian and a vector
        """
        dim = p.shape
        if len(dim) != 1 or dim[0] != self.m :
            raise ValueError('the vector p dimension should be ['+str(self.m)+' 1]')
        if self.m == 0 :
            return np.zeros(self.n, dtype=np.double)
        if isinstance(self.scale_con, np.ndarray):
            p = p.copy()
            p *= self.scale_con
        return self.lib.cutest_cjprod(self.n, self.m, x, p, 1)

    def __del__(self):
        """
        Delete problem
        """
        del(sys.modules[self.name])
        cmd = ['rm']+['-rf']+[self.directory]
        subprocess.call(cmd)


class QNCUTEstModel(QuasiNewtonModel, CUTEstModel):
    """CUTEst Model with a quasi-Newton Hessian approximation"""
    pass
