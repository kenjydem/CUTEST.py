import numpy as np
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class Steepest(object):

    """ 
    Class steepest provides a framework for solving unconstrained
    optimization problems by Steepest's method.
    """
    
    def __init__(self, model, x0 = None, etol=1.0e-6, itermax = 10000):
        
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
        self.dk = -self.g.copy() #/self.gNorm
        self.cos0 =  np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk))

        self.k = 0
        self.etol = etol
        self.itermax = itermax
    
    
    def solve(self):
        
        print"---------------------------------------"
        print "iter   f       ‖∇f‖    step    cosθ"
        print"---------------------------------------"
        print "%2d  %9.2e  %7.1e %6.4f %9.6f " % (self.k, self.f, self.gNorm, 0 ,self.cos0)
        while self.gNorm > self.etol and self.k < self.itermax:
            
            step = self.armijo()
            
            self.x = self.x + step * self.dk

            self.f = self.model.obj(self.x)
            self.g = self.model.grad(self.x)
            self.gNorm = np.linalg.norm(self.g)
            self.h = self.model.hess(self.x)
            self.dk = -self.g.copy()  #/self.gNorm
            self.cos0 = np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk)) 
            self.k += 1
            
            if (np.mod(self.k,10)==0):
                print"---------------------------------------"
                print "iter   f       ‖∇f‖    step    cosθ"
                print"---------------------------------------"
            print "%2d  %9.2e  %7.1e %6.4f %9.6f " % (self.k, self.f, self.gNorm, step,self.cos0)
        return self.x

    def armijo(self):
    
        xk = np.copy(self.x)
        fk = self.model.obj(xk)
        gk = self.model.grad(xk)
        slope = np.dot(gk, self.dk)  # Doit être < 0
        t = 1.0
        while self.model.obj(xk + t * self.dk) > fk + 1.0e-4 * t * slope:
            t /= 2#1.5 
        return t

