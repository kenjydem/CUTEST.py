import numpy as np
from pykrylov.linop.lbfgs import InverseLBFGSOperator as InvLBFGS
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class LBFGS(object):
    """
    Class LBFGSFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.
    """

    def __init__(self, model, x0 = None, itermax = 1000,etol = 1.0e-6,npair=5, save=False):
        
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
        self.save_data = save

    def search(self):

        print " k   f         ‖∇f‖    step "
        if self.save_data:
            result = " k   x           y           f  \n"
        while self.gNorm > self.etol and self.k < self.itermax:
                
            d = self.lbfgs.lbfgs_matvec(-self.g)
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.g,
                                         d,
                                         lambda t: self.model.obj(self.x+t*d),
                                         lambda t: self.model.grad(self.x+t*d),
                                         gtol= 0.1,
                                         ftol = 1.0e-4)
            SWLS.search()
            
            print "%2d  %9.2e  %7.1e  %7.1e" % (self.k, self.f, self.gNorm, SWLS.stp)
            
            if self.save_data:
                result += "%2d  %10.3e  %10.3e  %9.2e \n" % (self.k, self.x[0], self.x[1], self.f)


            self.x += SWLS.stp*d
            s = SWLS.stp*d         # Same than x_{k+1} - x_{k}
            New_gk = self.model.grad(self.x)
            y = New_gk - self.g
            self.lbfgs.store(s, y)
            
            self.f = self.model.obj(self.x)
            self.g = New_gk
            self.gNorm = np.linalg.norm(self.g)
            self.k += 1
        
        if self.save_data:
            result_file = open("result_LBFGS.txt" , "w")
            result_file.write(result)
            result_file.close

        return self.x
