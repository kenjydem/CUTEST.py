import numpy as np
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class Newton(object):

    """ 
    Class Newton provides a framework for solving unconstrained
    optimization problems by Newton's method.
    """
    
    def __init__(self, model, x0 = None, etol=1.0e-6, itermax = 1000, save = False):
        
        self.model = model
        if self.model.m > 0 :
            raise TypeError('This method only works on unconstrained problems')
        if x0 is None:
            x0 = np.copy(model.x0)
        self.x = x0
        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = np.linalg.norm(self.g)
        self.h = self.model.hess(self.x)
        self.k = 0
        self.etol = etol
        self.itermax = itermax

        self.save_data = save
    
    
    def search(self):
        
        print " k  f   ‖∇f‖"
        print "%2d  %7.1e %7.1e" % (self.k, self.f, self.gNorm)
        if self.save_data:
            result = " k   x           y           f  \n"
        while self.gNorm > self.etol and self.k < self.itermax:
            self.x -= np.linalg.solve(self.h, self.g)
            self.g = self.model.grad(self.x)
            self.gNorm = np.linalg.norm(self.g)
            self.h = self.model.hess(self.x)
            self.k += 1
            print "%2d  %7.1e %7.1e" % (self.k, self.f, self.gNorm)
            if self.save_data:
                result += "%2d  %10.3e  %10.3e  %9.2e \n" % (self.k, self.x[0], self.x[1], self.f)
        if self.save_data:
            result_file = open("result_Newton.txt" , "w")
            result_file.write(result)
            result_file.close
        
        return self.x

