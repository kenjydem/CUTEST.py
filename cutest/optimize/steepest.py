import numpy as np
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class Steepest(object):

    """ 
    Class steepest provides a framework for solving unconstrained
    optimization problems by Steepest's method.
    """
    
    def __init__(self, model, x0 = None, etol=1.0e-6, itermax = 10000, save = False):
        
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
        self.k = 0
        self.etol = etol
        self.itermax = itermax

        self.save_data = save
    
    
    def search(self):
        
        print " k  f   ‖∇f‖"
        print "%2d  %7.1e %7.1e" % (self.k, self.f, self.gNorm)
        if self.save_data:
            cos_angle = np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk))
            result1 = " k   f          g       x           y            \n"
            result2 = " k   f          g       x           y            \n"
            result3 = " k   f          g       x           y            \n"
            result1 += "%9.6f \n" %(self.x[0]) 
            result2 += "%9.6f \n" %(self.x[1]) 
            result3 += "%2d & %12.5e & %12.5e &  %9.6f & %9.6f & 0 & %9.6f \\\ \n" % (self.k,self.f, max(abs(self.g)), self.x[0], self.x[1],cos_angle)
        while self.gNorm > self.etol and self.k < self.itermax:
            
            step = self.armijo()
            
            self.x = self.x + step * self.dk

            self.f = self.model.obj(self.x)
            self.g = self.model.grad(self.x)
            self.gNorm = np.linalg.norm(self.g)
            self.h = self.model.hess(self.x)
            self.dk = -self.g.copy()  #/self.gNorm
            cos_angle = np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk)) 
            self.k += 1
            print "%2d  %7.1e %7.1e" % (self.k, self.f, self.gNorm)
            if self.save_data:
                result1 += "%9.6f \n" %(self.x[0])
                result2 += "%9.6f \n" %(self.x[1])
                result3 += "%2d & %12.5e & %12.5e &  %9.6f & %9.6f & %6.4f & %9.6f\\\ \n" % (self.k,self.f, max(abs(self.g)), self.x[0], self.x[1], step, cos_angle)
        if self.save_data:
            result_file = open("result_Steepest0.txt" , "w")
            result_file.write(result1)
            result_file.close
            result_file2 = open("result_Steepest1.txt" , "w")
            result_file2.write(result2)
            result_file2.close
            result_file3 = open("result_Steepest2.txt" , "w")
            result_file3.write(result3)
            result_file3.close

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

