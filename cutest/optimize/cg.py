import numpy as np
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class CG(object):
    """
    Class CGFramework provides a framework for solving unconstrained
    optimization problems by conjugate gradient with different search 
    lines methods
    """
    def __init__(self, model, x0=None, itermax=1000, etol=1.0e-5, strategy='HZ'):
        """
        Conjuguate gradient for non linear unconstraint problem 
        """

        self.model = model
        if self.model.m > 0 :
            raise TypeError('This method only works on unconstrained problems')
        
        if x0 is None:
            x0 = np.copy(model.x0)
        
        self.x = x0
        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = np.linalg.norm(self.g)
                                  
        self.strategy = strategy 
        self.p = -self.g.copy()

        self.k = 0
        self.etol = etol
        self.itermax = itermax
   

    def search(self, strategy= None):

        if strategy is not None:
            self.strategy = strategy

        print " k   f         ‖∇f‖      step      β "
        print "%2d  %9.2e  %7.1e          " % (self.k, self.f, self.gNorm)
        
        while self.gNorm > self.etol and self.k < self.itermax:
            
            # Search step with Strong Wolfe
            
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.g,
                                         self.p,
                                         lambda t: self.model.obj(self.x+t*self.p),
                                         lambda t: self.model.grad(self.x+t*self.p),
                                         gtol= 0.1,
                                         ftol = 1.0e-4)
            SWLS.search()
        
            #Seach line search
            self.x += SWLS.stp * self.p
        
            New_gk = self.model.grad(self.x)
            y = New_gk - self.g

            if self.strategy == 'HZ':
                bk = self.strategy_HZ(New_gk, y)
            elif self.strategy =='FR':
                bk = self.strategy_FR(New_gk)
            elif self.strategy == 'PR':
                bk = self.strategy_PR(New_gk, y)
            elif self.strategy == 'PR+':
                bk = self.strategy_PR_Plus(New_gk, y)
            elif self.strategy == 'PR-FR':
                bk = self.strategy_PR_FR(New_gk, y)

            self.p = -New_gk + bk * self.p
            self.f = self.model.obj(self.x)
            self.g = New_gk
            self.gNorm = np.linalg.norm(self.g)
            self.k += 1
                    
            print "%2d  %9.2e  %7.1e  %7.1e %7.1e" % (self.k, self.f, self.gNorm, SWLS.stp, bk)

        return self.x

    def strategy_FR(self, gk) :
        """
        Flectcher and Reeves strategy 
        """

        return np.dot(gk,gk)/np.dot(self.g, self.g)
   
    def strategy_PR(self, gk, yk):
        """
        Polak and Ribiere strategy
        """

        bk = np.dot(gk, yk)/np.dot(self.g,self.g)
        return bk
   
    def strategy_PR_Plus(self, gk, yk):
        """
        Polak and Ribiere + strategy
        """
            
        bk = self.strategy_PR(gk,yk)
        return max(bk,0)

    def strategy_PR_FR(self, gk, yk):
        """ Mixed between Polak and Ribiere
        and Flectcher and Reeves strategy """
              
        bk_PR = self.strategy_PR(gk,yk)
        bk_FR = self.strategy_FR(gk)
        
        #Check bk
        if bk_PR < -bk_FR:
            return -bk_FR
        elif abs(bk_PR) <= bk_FR:
            return bk_PR
        elif bk_PR > bk_FR:
            return bk_FR

    def strategy_HZ(self, gk, yk, n=0.01):
        """ 
        Hager and Zhang line search 
        """

        nk = -1./(np.linalg.norm(self.p)*min(n, np.linalg.norm(self.g)))
        pk_yk = np.dot(self.p,yk)
        bnk= 1./pk_yk * np.dot(yk - 2.*self.p*np.linalg.norm(yk)/pk_yk, gk)
        return max(bnk, nk)                            
