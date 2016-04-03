import numpy as np
from pykrylov.linop.lbfgs import InverseLBFGSOperator as InvLBFGS
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class LBFGS(object):
    """
    Class LBFGSFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.
    """

    def __init__(self, model, x0 = None, itermax = 1000,etol = 1.0e-6,npair=5):
        
        self.model = model
        if self.model.m > 0 :
            raise TypeError('This method only works on unconstrained problems')
        if x0 is None:
            x0 = np.copy(model.x0)
        self.x = x0
        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = np.linalg.norm(self.g)
            
        self.npair = npair
        self.lbfgs = InvLBFGS(self.model.n, self.npair)
            
        self.k = 0
        self.etol = etol
        self.itermax = itermax

    def search(self):

        print " k   fk        gNorm    t "
        print "%2d  %9.2e  %7.1e" % (self.k, self.f, self.gNorm)
        while self.gNorm > self.etol and self.k < self.itermax:
                
            d = self.lbfgs.lbfgs_matvec(-self.g)
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.g,
                                         d,
                                         lambda t: self.model.obj(self.x+t*d),
                                         lambda t: self.model.grad(self.x+t*d))
            SWLS.search()
            self.x += SWLS.stp*d
            s = SWLS.stp*d         # Same than x_{k+1} - x_{k}
            New_gk = self.model.grad(self.x)
            y = New_gk - self.g
            self.lbfgs.store(s, y)
            
            self.f = self.model.obj(self.x)
            self.g = New_gk
            self.gNorm = np.linalg.norm(self.g)
            self.k += 1

            print "%2d  %9.2e  %7.1e  %7.1e" % (self.k, self.f, self.gNorm, SWLS.stp)
        return self.x
