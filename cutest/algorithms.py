import numpy as np
from pykrylov.linop.lbfgs import InverseLBFGSOperator as InvLBFGS
from nlp.ls.pyswolfe import StrongWolfeLineSearch

class Newton(object):

    """ 
    Class Newton provides a framework for solving unconstrained
    optimization problems by Newton's method.
    """
    
    def __init__(self, model, x0 = None, etol=1.0e-6, itermax = 1000):
        
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
    
    
    def search(self):
        
        print " k  Normg   f"
        print "%2d  %7.1e %7.1e" % (self.k, self.gNorm, self.f)
        while self.gNorm > self.etol and self.k < self.itermax:
            self.x -= np.linalg.solve(self.h, self.g)
            self.g = self.model.grad(self.x)
            self.gNorm = np.linalg.norm(self.g)
            self.h = self.model.hess(self.x)
            self.k += 1
            print "%2d  %7.1e %7.1e" % (self.k, self.gNorm, self.f)
        return self.x

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

def ucg(self,maxiter=1000,etol=1.0e-5,ls_method='New'):
    """
    Conjuguate gradient for non linear unconstraint problem 
    """

    x = self.x0.copy()
    gk = self.grad(x)
    gNorm = np.linalg.norm(gk)
    fk = self.obj(x)
    k = 0
    pk = -gk
    print " k   fk        gNorm    t "
    print "%2d  %9.2e  %7.1e          " % (k, fk, gNorm)
    while gNorm > etol and k < maxiter:
            
        # Save tmp gradient value because it will be update in step search
        old_gk = gk.copy()

        # Search step with Strong Wolfe
        SWLS = StrongWolfeLineSearch(fk,
                                     gk,
                                     pk,
                                     lambda t: self.obj(x+t*pk),
                                     lambda t: self.grad(x+t*pk))
        # At this moment,gk is updated
        if SWLS.slope >0:
            SWLS = StrongWolfeLineSearch(fk,
                                         gk,
                                         -gk,
                                         lambda t: self.obj(x+t*pk),
                                         lambda t: self.grad(x+t*pk))
        SWLS.search()
        #ipdb.set_trace()
        #Seach line search
        x += SWLS.stp*pk
        y = gk - old_gk
        if ls_method is 'New':
            pk = self.new_line_search(gk, pk, y)
        elif ls_method is 'FR':
            pk = self.line_search_FR(gk, old_gk, pk)
        elif ls_method is 'PR':
            pk = self.line_search_PR(gk, old_gk, pk, y)
            
        fk = self.obj(x)
        gNorm = np.linalg.norm(gk)
        k += 1
                    
        #ipdb.set_trace()
        print "%2d  %9.2e  %7.1e  %7.1e" % (k, fk, gNorm, SWLS.stp)
    
    return x

def line_search_FR(self, gk, old_gk, pk):
    """ Flectcher and Reeves line search """

    bk = np.dot(gk,gk)/np.dot(old_gk,old_gk)
    return -gk + bk * pk
   
def line_search_PR(self, gk, old_gk, pk, yk):
    """ Polak and Ribiere linesearch
    Don't garanted direction descent"""

    bk = np.dot(gk, yk)/np.dot(old_gk,old_gk)
    New_bk = max(bk,0)
    return -gk + New_bk * pk

def new_line_search(self, gk, pk, yk, n=0.01):

    nk = -1./(norm(pk)*min(n, norm(gk)))
    pk_yk = np.dot(pk,yk)
    bnk= 1./pk_yk * np.dot(yk - 2.*pk*norm(yk)/pk_yk, gk)
    New_bnk = max(bnk, nk)                            
    return -gk + New_bnk * pk
    
    
    
def armijo (self, xk, dk, e=1.0e-4):
    """Methode d'armijio retournant une approximation de pas
    - xk : points d'evaluation (numpy.array)
    - dk : direction de descente (numpy.array)
    """

    fk = self.obj(xk)
    gk = self.grad(xk)
    slope = np.dot(gk, dk) #Doit etre < 0
    t = 1.0 #Initialisation du pas
    fki = self.obj(xk+t*dk)
    while fki > fk + e * t * slope:
        t /= 1.5
        fki = self.obj(xk+t*dk)
    return t
