import numpy as np
import logging
from nlp.ls.linesearch import ArmijoLineSearch
from nlp.model.linemodel import C1LineModel
from nlp.tools.exceptions import LineSearchFailure

class Steepest(object):

    """ 
    Class steepest provides a framework for solving unconstrained
    optimization problems by Steepest's method.
    """
    
    def __init__(self, model, **kwargs):
        
        self.model = model
        if self.model.m > 0 :
            raise TypeError('This method only works on unconstrained problems')
        
        self.x = kwargs.get("x0", np.copy(model.x0))
        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = np.linalg.norm(self.g)
        self.dk = -self.g.copy() 
        self.cos0 =  np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk))

        logger_name = kwargs.get("logger_name", "cutest.steepest")
        self.logger = logging.getLogger(logger_name)

        self.k = 0
        self.etol = kwargs.get("etol", 1.0e-5)
        self.itermax = kwargs.get("itermax", 10000)
    
    
    def solve(self):
        
        print"---------------------------------------"
        print "iter   f       ‖∇f‖    step    cosθ"
        print"---------------------------------------"
        print "%2d  %9.2e  %7.1e %6.4f %9.6f " % (self.k, self.f, self.gNorm, 0 ,self.cos0)
        
        while self.gNorm > self.etol and self.k < self.itermax:
            
            line_model = C1LineModel(self.model,self.x,self.dk)
            step0 = max(1.0e-3, 1.0 / self.gNorm) if self.k == 0 else 1.0
            ls = ArmijoLineSearch(line_model,step= step0)
            
            try:
                for step in ls:
                    self.logger.debug("step=%6.2e, f=%6.2e", step, ls.trial_value)
            except LineSearchFailure:
                continue

            self.x = ls.iterate
            self.f = self.model.obj(self.x)
            self.g = self.model.grad(self.x)
            self.gNorm = np.linalg.norm(self.g)
            self.dk = -self.g.copy()  #/self.gNorm
            self.cos0 = np.dot(self.g,self.dk)/(self.gNorm*np.linalg.norm(self.dk)) 
            self.k += 1
            
            if (np.mod(self.k,10)==0):
                print"---------------------------------------"
                print "iter   f       ‖∇f‖    step    cosθ"
                print"---------------------------------------"
            print "%2d  %9.2e  %7.1e %6.4f %9.6f " % (self.k, self.f, self.gNorm, ls.step,self.cos0)
        return self.x
