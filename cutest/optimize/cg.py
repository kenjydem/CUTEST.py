import numpy as np
from nlp.ls.pyswolfe import StrongWolfeLineSearch

def CG(self,maxiter=1000,etol=1.0e-5,ls_method='New'):
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
