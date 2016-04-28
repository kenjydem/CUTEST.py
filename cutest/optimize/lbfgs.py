import numpy as np
from pykrylov.linop.lbfgs import InverseLBFGSOperator as InvLBFGS
from nlp.ls.pymswolfe import StrongWolfeLineSearch

class LBFGS(object):
    """
    Class LBFGSFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.
    """

    def __init__(self, model, **kwargs):
        
        self.model = model
        if self.model.m > 0 :
            raise TypeError('This method only works on unconstrained problems')
        
        self.x = kwargs.get("x0", np.copy(model.x0))
        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = np.linalg.norm(self.g)
        self.npair = kwargs.get("npair", 5)
        self.lbfgs = InvLBFGS(self.model.n, self.npair)

        self.d = self.lbfgs.lbfgs_matvec(-self.g)
        self.cos0 =  np.dot(self.g,self.d)/(self.gNorm*np.linalg.norm(self.d))
            
            
        self.k = 0
        self.etol = kwargs.get("etol", 1.0e-5)
        self.itermax = kwargs.get("itermax", 10000)

    def solve(self):

        while self.gNorm > self.etol and self.k < self.itermax:
                
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.x,
                                         self.g,
                                         self.d,
                                         lambda t: self.model.obj(t),
                                         lambda t: self.model.grad(t),
                                         gtol= 0.1,
                                         ftol = 1.0e-4)
            SWLS.search()
            
            if (np.mod(self.k,10)==0):
                print"---------------------------------------"
                print "iter   f       ‖∇f‖    step    cosθ"
                print"---------------------------------------"
            print "%2d  %9.2e  %7.1e %6.4f %9.6f " % (self.k, self.f, self.gNorm, SWLS.stp,self.cos0)

            self.x += SWLS.stp*self.d
            s = SWLS.stp*self.d         # Same than x_{k+1} - x_{k}
            New_gk = self.model.grad(self.x)
            y = New_gk - self.g
            self.lbfgs.store(s, y)
            
            self.f = self.model.obj(self.x)
            self.g = New_gk
            self.gNorm = np.linalg.norm(self.g)
            self.d = self.lbfgs.lbfgs_matvec(-self.g)
            self.cos0 =  np.dot(self.g,self.d)/(self.gNorm*np.linalg.norm(self.d))
            self.k += 1

        return self.x
