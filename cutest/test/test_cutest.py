from cutest.model import cutestmodel
import numpy as np
import scipy.sparse as sparse

def status(bool):
    if bool:
        print 'OK'
    else :
        print 'KO' 


name='HS13'

if name[-4:] == ".SIF":
    name = name[:-4]
        
print 'Reading problem ' + name + '...'
try:
    prob = cutestmodel.CUTEstModel(name)
    x = prob.x0
    

    print 'Comparison of grad and sgrad...'
    try:
        g = prob.grad(x)
        sg = prob.sgrad(x)
        status(np.array_equal(g, np.resize(sg.todense(), g.shape)))
    except:
        pass
    
    print 'Test of cons and icons...'
    try:
        c = prob.cons(x)
        ic = np.zeros(prob.m)
        for i in range(prob.m):
            ic[i] = prob.icons(i+1, x)
        status(np.array_equal(c, ic))
    except:
        pass

    print 'Test of igrad and sigrad...'
    try:
        ig = prob.igrad(1, x)
        sig = prob.sigrad(1, x)
        status(np.array_equal(ig, np.resize(sig.todense(), ig.shape)))
    except:
        pass

    print 'Test of jac and igrad...'
    try:
        j = prob.jac(x)
        jj = np.zeros(j.shape)
        for i in range(prob.m):
            jj[i,] = prob.igrad(i+1, x)
        status(np.array_equal(j,jj))
    except:
        pass

    print 'Test of hess et shess...'
    try:
        z = np.ones((prob.m,))
        h = prob.hess(x, z)
        sh = prob.shess(x, z)
        status(np.array_equal(h, np.resize(sh.todense(), h.shape)))
    except:
        pass

    print 'Test of ihess et ishess...'
    try:
        ih = prob.ihess(x, 1)
        sih = prob.ishess(x, 1)
        status(np.array_equal(ih, np.resize(sih.todense(), ih.shape)))
    except:
        pass

    print 'Test of jprod ...'
    
    try:
        status(np.array_equal(np.dot(j, np.ones(prob.n)),
                      prob.jprod(x, np.ones(prob.n))))
    except:
        pass

    print 'Test of jtprod...'
    try:
        status(np.array_equal(np.dot(j.transpose(), np.ones(prob.m)),
                              prob.jtprod(x, np.ones(prob.m))))
    except:
        pass

    print 'Test of hprod...'
    try:
        status(np.array_equal(np.dot(h, 2*np.ones(prob.n)),
                          prob.hprod(x, z, 2*np.ones(prob.n))))
    except:
        pass

    print 'Test of hiprod...'
    try:
        status(np.array_equal(np.dot(ih, 2*np.ones(prob.n)),
                        prob.hiprod(1, x, 2*np.ones(prob.n))))
    except:
        pass
except:
    raise ImportError('Impossible to load the problem check if CUTEst is install and try again')
